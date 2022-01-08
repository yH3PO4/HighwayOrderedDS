import typing
import math
import json
import urllib.parse

from dataclasses import dataclass

import fire
import pandas as pd
import geopandas as gpd
import networkx as nx

from shapely.geometry import Point, LineString

import plot

pd.set_option("display.unicode.east_asian_width", True)


@dataclass
class RoadName:
    main_name: str
    sub_name: typing.Optional[str] = None

    def __str__(self):
        if self.sub_name is None:
            return self.main_name
        else:
            return f"{self.main_name}_{self.sub_name}"


class WikipediaParser:
    MAX_SEARCH = 25

    def __init__(self,
                 url: str,
                 table_num: typing.Optional[int] = None) -> None:
        self.url = url
        self.table_num = table_num

    def fetch_table(self) -> pd.DataFrame:
        """
        wikipediaから施設名とキロポスト情報を取得する。

        Returns:
            施設名とキロポスト情報のDataframe
        """

        dfs = pd.read_html(self.url, match="施設名")

        if self.table_num is None:
            for i in range(WikipediaParser.MAX_SEARCH):
                try:
                    df = self._format_table(dfs[i])
                    print(df)
                    s = input("この表で正しいですか？ [Y/N]:").lower()
                    if s in ("n", "no"):
                        continue
                    else:
                        return df
                except IndexError:
                    continue
        else:
            try:
                df = WikipediaParser._format_table(dfs[self.table_num])
                return df
            except IndexError:
                raise RuntimeError(f"指定された{self.table_num}番目の表が見つかりませんでした。")
        raise RuntimeError("Wikipediaからの表の取得に失敗しました。")

    @staticmethod
    def _format_table(df: pd.DataFrame) -> pd.DataFrame:
        """
        wikipediaの表を整形し、必要な列だけ返す。
        Args:
            df: wikipediaの表のDataframe

        Returns:
            施設名とキロポスト情報のDataframe
        """
        df.columns = pd.Index(df.columns)  # MultiIndexだったらIndexに直す
        df.rename(columns={df.columns[1]: "name", df.columns[3]: "kp"}, inplace=True)
        df = df[["name", "kp"]].copy()
        df["kp"] = pd.to_numeric(df["kp"], errors="coerce")
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

        if df.empty:
            raise ValueError

        return df


class RoadMaker:
    HIGHWAY_SECTION = r".\data\N06-20_HighwaySection_fixed.geojson"
    JOINT = r".\data\N06-20_Joint_fixed.geojson"
    KP = r".\kp.csv"
    NODELINK = r".\data\nodelink.json"
    LENGTH = r".\data\length.json"
    RESULT_HIGHWAY_POINT = r".\data\highway_point.geojson"
    RESULT_HIGHWAY_PATH = r".\data\highway_path.geojson"

    def __init__(self,
                 road_name: RoadName,
                 update_graph: bool = False,
                 local: bool = False,
                 local_csv: bool = False,
                 table_search: bool = True,
                 wiki_table_num: int = 0) -> None:
        self.road_name = RoadName(*road_name.split("_", 1))  # "_" 以降は支線名として登録
        self.update_graph = update_graph
        self.local = local
        self.local_csv = local_csv
        self.table_search = table_search
        self.wiki_table_num = wiki_table_num

        if update_graph:
            self.G = RoadMaker._section_to_nodelink()
        else:
            self.G = RoadMaker._read_nodelink_json()

        self.df_wiki_table = pd.DataFrame()
        self.df_point = pd.DataFrame()
        self.df_path = pd.DataFrame()
        self.df_joint = RoadMaker._read_joint_json()

    def fetch_joints(self) -> None:
        """
        道路について、施設名・キロポスト情報・座標情報のテーブルを渡す。
        名前が重複したら選択して削除可能にする機能もここ
        """
        if self.local_csv:
            self.df_wiki_table = pd.read_csv(RoadMaker.KP, index_col=0)
        else:
            if self.local:
                target_url = rf".\wikipedia\{self.road_name.main_name} - Wikipedia.html"
            else:
                target_url = rf"https://ja.wikipedia.org/wiki/{urllib.parse.quote(self.road_name.main_name)}"

            wp = WikipediaParser(target_url, (None if self.table_search else self.wiki_table_num))
            self.df_wiki_table = wp.fetch_table()
            self.df_wiki_table.to_csv(RoadMaker.KP, encoding="utf-8-sig")

        self.df_wiki_table = self._parse_attr(self.df_wiki_table)

        self.df_wiki_table.drop_duplicates(subset=["name", "is_IC", "is_JCT", "is_SIC", "is_SAPA"],
                                           keep="first", inplace=True)  # Wikipediaがたまに多段になってるのを消す

        self.df_point = pd.merge(self.df_wiki_table, self.df_joint, on="name", how="left")
        print(self.df_point)

        # 名称の表記ゆれに対応
        df_nan = self.df_point[self.df_point["coordinates"].isnull()].copy()
        df_nan.drop("coordinates", axis=1, inplace=True)

        if not df_nan.empty:
            print("高速道路時系列データと対応がない施設があります。")
            df_nan["name_prefix"] = df_nan["name"].replace(r"[ -~]+", "", regex=True)
            df_prefix = pd.merge(df_nan, self.df_joint, on="name_prefix", how="left", suffixes=["", "_joint"])
            print(df_prefix.name[
                      (df_prefix["coordinates"].isnull()) & (
                              df_prefix["is_IC"] | df_prefix["is_SIC"] | df_prefix["is_JCT"])])

            print("名称の一部が高速道路時系列データと合致する施設があります。")
            print(df_prefix[["name", "name_prefix", "name_joint", "coordinates"]].dropna(subset=["coordinates"]))
            remain_indices = set(map(int, input("残したい行番号を一行に続けて入力してください:").split()))
            df_prefix.drop(set(df_prefix.index) - remain_indices, inplace=True)
            del df_prefix["kp"]

            self.df_point = self.df_point.merge(df_prefix[["name", "name_prefix", "coordinates"]], on="name",
                                                how="left")
            self.df_point["coordinates"] = self.df_point["coordinates_x"].combine_first(self.df_point["coordinates_y"])
            self.df_point.dropna(subset=["coordinates"], inplace=True)
            self.df_point.drop(["name_prefix_x", "name_prefix_y", "coordinates_x", "coordinates_y"], axis=1,
                               inplace=True)
            self.df_point.reset_index(inplace=True, drop=True)
            print(self.df_point)

        # 名前が重複したら選択して削除可能にする
        dup = self.df_point[self.df_point.duplicated(subset="name", keep=False)]
        if not dup.empty:
            print("施設名の重複があります。")
            print(dup)
            plot.plot_points(dup)
            remain_indices = set(map(int, input("残したい行番号を一行に続けて入力してください:").split()))
            self.df_point.drop(set(dup.index) - remain_indices, inplace=True)
            self.df_point.reset_index(inplace=True, drop=True)
            print(self.df_point)

        # 地点の最終確認
        plot.plot_points(self.df_point)
        del_indices = set(map(int, input("削除したい行番号を一行に続けて入力してください:").split()))
        self.df_point.drop(del_indices, inplace=True)
        self.df_point.reset_index(inplace=True, drop=True)

    def find_path(self) -> None:
        """
        最短経路探索によって各施設間の経路を決定する。
        """
        data_path = {"source": [], "target": [], "path": []}
        for i in range(len(self.df_point) - 1):
            source, target = tuple(self.df_point.loc[i, "coordinates"]), tuple(self.df_point.loc[i + 1, "coordinates"])
            try:
                path = nx.astar_path(self.G, source, target, RoadMaker._substitution_distance)
                data_path["source"].append(self.df_point.loc[i, "name"])
                data_path["target"].append(self.df_point.loc[i + 1, "name"])
                data_path["path"].append(path)
            except nx.NetworkXNoPath:
                continue
        self.df_path = pd.DataFrame(data_path)
        plot.plot_oneline(self)

    def calc_line_length(self) -> None:
        """
        道路の開業済み延長を計算して `length.json` に登録する。`
        """
        L = abs(pd.merge(self.df_path["target"], self.df_point[["name", "kp"]],
                         left_on="target", right_on="name")["kp"] -
                pd.merge(self.df_path["source"], self.df_point[["name", "kp"]],
                         left_on="source", right_on="name")["kp"]).sum()

        print(f"開業済み延長: {round(L, 1)} km")

        with open(RoadMaker.LENGTH, "r", encoding="utf-8-sig") as f:
            d = json.load(f)

        d[str(self.road_name)] = round(L, 1)

        with open(RoadMaker.LENGTH, 'w', encoding="utf-8-sig") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)

    def delete_wrong_path(self) -> None:
        """
        間違って推定された区間がなくなるまで区間の削除を繰り返す
        """
        while True:
            del_indices = set(map(int, input("削除したい区間の番号を一行に続けて入力してください:").split()))
            if not del_indices:
                return
            self._del_path(del_indices)

    def add_path(self) -> None:
        """
        区間を追加する。
        主に環状線で終点→起点の区間を追加する場合に使用。
        """
        while True:
            add_pair = tuple(map(int, input("追加したい区間の両端にあたる2地点の番号を入力してください:").split()))
            if not add_pair:
                return
            if len(add_pair) != 2:
                print("入力されるのは2つの整数である必要があります。")
                continue
            source, target = add_pair
            self._add_path(source, target)

    def estimate_SAPA(self) -> None:
        """
        時系列データに存在しないSA/PAの位置を推定し、Dataframeに追加する。
        """
        df_SAPA = pd.merge(self.df_point[["name", "coordinates"]], self.df_wiki_table,
                           on="name", how="right", indicator=True)

        # SA, PA 以外は今のところ扱わない
        df_SAPA = df_SAPA[(df_SAPA["_merge"] == "both") | df_SAPA["is_SAPA"]].copy()
        df_SAPA["name"].where(df_SAPA["_merge"] == "both",
                              df_SAPA["name"].replace(r"(.*?(SA|PA)).*", r"\1", regex=True),
                              inplace=True)
        df_SAPA.drop_duplicates(subset=["name", "kp"], inplace=True)

        # 存在しないSA,PAを削除
        if not (estimate := df_SAPA[df_SAPA["_merge"] == "right_only"]).empty:
            print("時系列データにない SA, PA が見つかりました。")
            print(estimate.name)
            del_indices = set(map(int, input("削除したい行番号を一行に続けて入力してください:").split()))
            df_SAPA.drop(del_indices, inplace=True)
        df_SAPA.reset_index(drop=True, inplace=True)

        df_SAPA = df_SAPA.merge(self.df_path, left_on="name", right_on="source", how="left")

        data_path = {"source": [], "target": [], "path": []}

        for i in range(len(df_SAPA) - 1):
            if df_SAPA.at[i + 1, "_merge"] == "right_only":
                dist = abs(df_SAPA.at[i, "kp"] - df_SAPA.at[i + 1, "kp"])
                path = df_SAPA.at[i, "path"]
                j = 0
                while dist > 0:
                    dist -= RoadMaker._euclidean_distance(path[j], path[j + 1])
                    j += 1
                    if j == len(path) - 2:
                        print(f"{df_SAPA.at[i + 1, 'name']}の推定された位置と{df_SAPA.at[i + 2, 'name']}の位置が近接しています。")
                        break

                new_coordinate = path[j]
                new_path = path[j:]
                path = path[:j + 1]

                df_SAPA.at[i, "target"] = df_SAPA.at[i + 1, "name"]
                df_SAPA.at[i + 1, "source"] = df_SAPA.at[i + 1, "name"]
                df_SAPA.at[i + 1, "target"] = df_SAPA.at[i + 2, "name"]  # 末端がSAでないことを利用
                df_SAPA.at[i + 1, "coordinates"] = new_coordinate
                df_SAPA.at[i, "path"] = path
                df_SAPA.at[i + 1, "path"] = new_path

            if isinstance(df_SAPA.at[i, "path"], list):
                data_path["source"].append(df_SAPA.loc[i, "source"])
                data_path["target"].append(df_SAPA.loc[i, "target"])
                data_path["path"].append(df_SAPA.loc[i, "path"])

        self.df_point = df_SAPA[["name", "coordinates", "kp", "is_IC", "is_SIC", "is_JCT", "is_SAPA"]]
        self.df_path = pd.DataFrame(data_path)
        plot.plot_oneline(self)

    def to_geojson(self) -> None:
        """
        2つのgeojsonファイルに出力する。
        """
        # Point
        gdf_point: gpd.GeoDataFrame = gpd.read_file(RoadMaker.RESULT_HIGHWAY_POINT)
        if not gdf_point.empty:
            gdf_point = gdf_point[gdf_point["road_name"] != str(self.road_name)].copy()  # もとのを削除

        gdf_point_cur = gpd.GeoDataFrame(self.df_point,
                                         geometry=self.df_point["coordinates"].apply(lambda x: Point(*x)))
        gdf_point_cur.drop(columns="coordinates", inplace=True)
        gdf_point_cur["order"] = pd.RangeIndex(stop=len(gdf_point_cur))
        gdf_point_cur["road_name"] = str(self.road_name)
        gdf_point_cur["is_IC"] = gdf_point_cur["is_IC"].astype(int)
        gdf_point_cur["is_SIC"] = gdf_point_cur["is_SIC"].astype(int)
        gdf_point_cur["is_JCT"] = gdf_point_cur["is_JCT"].astype(int)
        gdf_point_cur["is_SAPA"] = gdf_point_cur["is_SAPA"].astype(int)
        gdf_point_cur = gdf_point_cur.reindex(
            columns=["road_name", "order", "name", "kp", "is_IC", "is_SIC", "is_JCT", "is_SAPA", "geometry"])

        print(gdf_point_cur)
        gdf_point = gpd.GeoDataFrame(pd.concat([gdf_point, gdf_point_cur]))
        gdf_point.to_file(RoadMaker.RESULT_HIGHWAY_POINT, driver='GeoJSON')

        # LINESTRING
        gdf_path: gpd.GeoDataFrame = gpd.read_file(RoadMaker.RESULT_HIGHWAY_PATH)
        if not gdf_path.empty:
            gdf_path = gdf_path[gdf_path["road_name"] != str(self.road_name)].copy()  # もとのを削除

        gdf_path_cur = gpd.GeoDataFrame(self.df_path[["source", "target"]],
                                        geometry=self.df_path["path"].apply(lambda x: LineString(x)))
        gdf_path_cur["order"] = pd.RangeIndex(stop=len(gdf_path_cur))
        gdf_path_cur["road_name"] = str(self.road_name)
        gdf_path_cur = gdf_path_cur.reindex(
            columns=["road_name", "order", "source", "target", "geometry"])

        print(gdf_path_cur)
        gdf_path = gpd.GeoDataFrame(pd.concat([gdf_path, gdf_path_cur]))
        gdf_path.to_file(RoadMaker.RESULT_HIGHWAY_PATH, driver='GeoJSON')

    def _del_path(self, del_indices: set[int]) -> None:
        """
        余計な区間を個別に削除する。
        Args:
            del_indices(set): 削除する区間の始点
        """
        self.df_path.drop(del_indices, inplace=True)
        self.df_path.reset_index(inplace=True)
        self.calc_line_length()
        plot.plot_oneline(self)
        print(f"区間 {del_indices} が削除されました。")

    def _add_path(self, source_idx: int, target_idx: int) -> None:
        """
        新しい区間を追加する。
        """
        source: tuple[float, float] = tuple(self.df_point.loc[source_idx, "coordinates"])
        target: tuple[float, float] = tuple(self.df_point.loc[target_idx, "coordinates"])
        path = nx.astar_path(self.G, source, target, RoadMaker._substitution_distance)
        sr = pd.Series([self.df_point.loc[source_idx, "name"],
                        self.df_point.loc[target_idx, "name"], path], index=self.df_path.columns)
        self.df_path = self.df_path.append(sr, ignore_index=True)
        self.calc_line_length()
        plot.plot_oneline(self)
        print(f"{source_idx}, {target_idx} をつなぐ区間が追加されました。")

    @staticmethod
    def _euclidean_distance(l1: tuple[float, float], l2: tuple[float, float]) -> float:
        """
        緯度経度から比較的正確な距離を導出する。
        Args:
            l1: 座標(緯度, 経度)
            l2: 座標(緯度, 経度)

        Returns:
            二点間の距離(km単位)
        """
        lat1 = l1[0] * math.pi / 180
        lng1 = l1[1] * math.pi / 180
        lat2 = l2[0] * math.pi / 180
        lng2 = l2[1] * math.pi / 180
        return 6371 * math.acos(
            math.cos(lat1) * math.cos(lat2) * math.cos(lng2 - lng1) + math.sin(lat1) * math.sin(lat2))

    @staticmethod
    def _substitution_distance(l1: tuple[float, float], l2: tuple[float, float]) -> float:
        """
        `astar` の距離関数に用いる、おおよそ距離に比例する値を求める。
        Args:
            l1: 座標(緯度, 経度)
            l2: 座標(緯度, 経度)

        Returns:
            二点間の距離におおよそ比例する値
        Notes:
            `euclidean_distance` を距離関数に使うと誤差で壊れるようなのでやむなくこれで代用している。
        """
        return ((l1[0] - l2[0]) ** 2 + (l1[1] - l2[1]) ** 2) ** (1 / 2)

    @staticmethod
    def _parse_attr(df: pd.DataFrame) -> pd.DataFrame:
        """
        施設名から「ICであるかどうか」「SA/PAであるかどうか」などの属性を判定してから施設名を修正する。
        """
        df["is_IC"] = df["name"].str.contains("[^S]+IC|出入口|出口|入口", regex=True)
        df["is_SIC"] = df["name"].str.contains("SIC")
        df["is_JCT"] = df["name"].str.contains("JCT")
        df["is_SAPA"] = df["name"].str.contains("SA|PA", regex=True)

        df["name"].replace(r"(.*)出入口$", r"\1", regex=True, inplace=True)
        df["name"].replace(r"(.*)(出口|入口)$", r"\1", regex=True, inplace=True)
        df["name"].replace(r"([^S]+)IC.*", r"\1", regex=True, inplace=True)  # SICは残す
        df["name"].replace(r"(.*?(JCT|SIC)).*", r"\1", regex=True, inplace=True)

        return df

    @staticmethod
    def _read_joint_json() -> pd.DataFrame:
        """
        N06-18_Joint.jsonのデータを整形して渡す
        Returns:
            施設名と座標情報のDataframe
        """
        with open(RoadMaker.JOINT, encoding="utf-8_sig") as f:
            data = json.load(f)
        df_joint: pd.DataFrame = pd.json_normalize(data["features"])
        df_joint = df_joint[df_joint["properties.N06_014"] == 9999]
        df_joint.rename(columns={"properties.N06_018": "name", "geometry.coordinates": "coordinates"}, inplace=True)
        df_joint.replace(r"\\/", r"\/", regex=True, inplace=True)
        df_joint["name_prefix"] = df_joint["name"].replace(r"[ -~]+", "", regex=True)
        return df_joint[["name", "name_prefix", "coordinates"]]

    @staticmethod
    def _section_to_nodelink() -> nx.Graph:
        """
        `N06-18_HighwaySection.json`のデータをnx.Graphに整形し、`nodelink.json`に保存する。
        Returns:
            高速道路のGraph
        """
        with open(RoadMaker.HIGHWAY_SECTION, encoding="utf-8_sig") as f:
            data = json.load(f)
        df_section: pd.DataFrame = pd.json_normalize(data["features"])
        df_section.rename(columns={"geometry.coordinates": "coordinates_edge"}, inplace=True)

        G = nx.Graph()
        for edges in df_section["coordinates_edge"]:
            for i in range(len(edges) - 1):
                G.add_edge(tuple(edges[i]), tuple(edges[i + 1]),
                           weight=RoadMaker._euclidean_distance(edges[i], edges[i + 1]))

        with open(RoadMaker.NODELINK, "w", encoding="utf-8_sig") as f:
            json.dump(nx.node_link_data(G), f, ensure_ascii=False)

        print("nodelink.json を更新しました。")
        return G

    @staticmethod
    def _read_nodelink_json() -> nx.Graph:
        """
        `nodelink.json`のデータをnx.Graphにして返す
        Returns:
            高速道路のGraph
        """

        with open(RoadMaker.NODELINK, encoding="utf-8-sig") as f:
            G: nx.Graph = nx.node_link_graph(json.load(f))
        return G


def main(road_name: RoadName,
         update_graph: bool = False,
         local: bool = False,
         local_csv: bool = False,
         table_search: bool = True,
         wiki_table_num: int = 0) -> None:
    rm = RoadMaker(road_name, update_graph, local, local_csv, table_search, wiki_table_num)
    rm.fetch_joints()
    rm.find_path()
    rm.calc_line_length()
    rm.delete_wrong_path()
    rm.estimate_SAPA()
    rm.add_path()
    rm.to_geojson()


if __name__ == "__main__":
    fire.Fire(main)
