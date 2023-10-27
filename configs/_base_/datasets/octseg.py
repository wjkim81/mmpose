dataset_info = dict(
    dataset_name='octseg',
    keypoint_info={
        0:
        dict(
            name='ostium_point_start',
            id=0,
            color=[51, 153, 255],
            type='',
            swap='ostium_point_end'),
        1:
        dict(
            name='ostium_point_end',
            id=1,
            color=[51, 153, 255],
            type='',
            swap='ostium_point_start')
    },
    skeleton_info={
        0:
        dict(link=('ostium_point_start', 'ostium_point_end'), id=0, color=[0, 255, 0])
    },
    joint_weights=[
        1., 1.
    ],
    sigmas=[
        1., 1.
    ])