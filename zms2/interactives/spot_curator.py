import napari
import pickle


default_spot_opacity = 0.3
spot_layer_number = 2   # update if more layers are added
opacity_increments = 0.1


def spot_curator(spot_im, spots, nuc_im=None, spots_dir=r'./', spots_name='spots'):
    # launch the viewer with spot image
    viewer = napari.view_image(spot_im, colormap='green', contrast_limits=(0, 4000))

    # if nuclear image is provided, add it
    if nuc_im is not None:
        viewer.add_image(nuc_im, colormap='magenta', opacity=0.5, contrast_limits=(0., 1.))

    # display first spot
    spot_id = 0
    points_layer = viewer.add_points(spots[spot_id]['com'], name='spot' + str(spot_id) + str(spots[spot_id]['kept']), opacity=default_spot_opacity)
    viewer.dims.set_point(axis=0, value=spots[spot_id]['com'][0])

    """key commands"""
    @viewer.bind_key('.')
    def next_spot(viewer):
        nonlocal spot_id
        viewer.layers.remove('spot' + str(spot_id) + str(spots[spot_id]['kept']))
        spot_id = spot_id + 1
        if len(spots) - 1 >= spot_id >= 0:
            viewer.add_points(spots[spot_id]['com'], name='spot' + str(spot_id) + str(spots[spot_id]['kept']), opacity=default_spot_opacity)
            viewer.dims.set_point(axis=0, value=spots[spot_id]['com'][0])
        else:
            viewer.status = 'outside range of spot numbers!'

    @viewer.bind_key('d')
    def alt_next_spot(viewer):
        nonlocal spot_id
        viewer.layers.remove('spot' + str(spot_id) + str(spots[spot_id]['kept']))
        spot_id = spot_id + 1
        if len(spots) - 1 >= spot_id >= 0:
            viewer.add_points(spots[spot_id]['com'], name='spot' + str(spot_id) + str(spots[spot_id]['kept']),
                              opacity=default_spot_opacity)
            viewer.dims.set_point(axis=0, value=spots[spot_id]['com'][0])
        else:
            viewer.status = 'outside range of spot numbers!'

    @viewer.bind_key(',')
    def previous_spot(viewer):
        nonlocal spot_id
        viewer.layers.remove('spot' + str(spot_id) + str(spots[spot_id]['kept']))
        spot_id = spot_id - 1
        if len(spots) - 1 >= spot_id >= 0:
            viewer.add_points(spots[spot_id]['com'], name='spot' + str(spot_id) + str(spots[spot_id]['kept']), opacity=default_spot_opacity)
            viewer.dims.set_point(axis=0, value=spots[spot_id]['com'][0])
        else:
            viewer.status = 'outside range of spot numbers!'

    @viewer.bind_key('t')
    def true_spot(viewer):
        nonlocal spot_id
        nonlocal spots
        spots[spot_id]['kept'] = True
        viewer.layers[spot_layer_number].name = 'spot' + str(spot_id) + str(spots[spot_id]['kept'])

    @viewer.bind_key('f')
    def false_spot(viewer):
        nonlocal spot_id
        nonlocal spots
        spots[spot_id]['kept'] = False
        viewer.layers[spot_layer_number].name = 'spot' + str(spot_id) + str(spots[spot_id]['kept'])

    @viewer.bind_key('s')
    def save_spots(viewer):
        nonlocal spots
        with open(spots_dir + '/' + spots_name, "wb") as fp:
            pickle.dump(spots, fp)
        viewer.status = 'saved spots'

    @viewer.bind_key('r')
    def increase_spot_opacity(viewer):
        current_opacity = viewer.layers[spot_layer_number].opacity
        if current_opacity + opacity_increments <= 1.0:
            viewer.layers[spot_layer_number].opacity += opacity_increments
        else:
            viewer.layers[spot_layer_number].opacity = 1.0

    @viewer.bind_key('e')
    def decrease_spot_opacity(viewer):
        current_opacity = viewer.layers[spot_layer_number].opacity
        if current_opacity - opacity_increments >= 0.0:
            viewer.layers[spot_layer_number].opacity -= opacity_increments
        else:
            viewer.layers[spot_layer_number].opacity = 0.0

    napari.run()

    return spots, viewer
