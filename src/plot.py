import folium


def gen_map():
    """
    マップの基本部分を生成
    Returns:
        m(folium.Map)
    """
    copyright_st = '電子国土基本図'

    m = folium.Map(location=[35.681167, 139.767052],
                   zoom_start=5,
                   attr=copyright_st,
                   tiles='https://cyberjapandata.gsi.go.jp/xyz/pale/{z}/{x}/{y}.png')
    return m


def plot_points(df):
    """
    施設座標のマーカーをプロット
    Args:
        rm: RoadMakerクラス
    """
    m = gen_map()
    for feature in df.itertuples():
        folium.Marker((feature.coordinates[1], feature.coordinates[0]),
                      popup=f"{feature.Index, feature.name}\n{feature.coordinates}").add_to(m)
    m.save("plot_admin.html")


def plot_oneline(rm):
    """
    路線別で区間の曲線と施設座標のマーカーをプロット
    Args:
        rm: RoadMakerクラス
    """
    m = gen_map()

    for feature in rm.df_point.itertuples():
        folium.Marker((feature.coordinates[1], feature.coordinates[0]),
                      popup=f"{feature.Index, feature.name}\n{feature.coordinates}").add_to(m)
    for feature in rm.df_path.itertuples():
        folium.PolyLine(locations=[(x[1], x[0]) for x in feature.path],
                        popup=f"{feature.Index}",
                        color="green").add_to(m)
    m.save("plot_admin.html")
