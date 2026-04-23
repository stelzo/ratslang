registration {
    map {
        # @recommended
        # @type = length
        # @description = "Feature size used for registration"
        resolution = 50cm

        # @recommended
        # @type = number
        # @description = "How many Voxels of the previous defined resolution to save at any time"
        capacity = 131072

        # @type = number
        # @description = "Magic weighting value for handling the sensivity to new scans"
        distribution_update_sensitivity = 10.0
    }

    # @type = number
    # @description = "Maximum IEKF correction iterations per scan"
    max_iter = 4

    # @type = length
    # @description = "Maximum distance to any neighbor"
    neighbors_max_range = 5m
}

preprocessing {
    downsample {
        # @recommended
        # @type = boolean
        # @description = "Output the full undistorted cloud on the cloud output topic"
        bypass_output_downsample = true

        # @recommended
        # @type = number
        # @description = "Downsample the incoming scan by keeping only every n-th point"
        point_skip = 3

        # @type = number
        # @description = "Estimated sizes of features in the incoming scan"
        query_resolution_factor = 1.0
    }

    # @type = number
    # @description = "Assume the first n scans are recorded while standing still"
    initial_scan_accumulate = 3

    filter {
        bounding_box {
            # @type = range<length>
            # @description = "Bounding Box x range"
            x = ..

            # @type = range<length>
            # @description = "Bounding Box y range"
            y = ..

            # @type = range<length>
            # @description = "Bounding Box z range"
            z = ..

            # @type = boolean
            # @description = "If true, we only accept points inside this bounding_box"
            invert = false
        }

        distance {
            # @type = range<length>
            # @description = "Distance range filter"
            range = 4m..

            # @type = boolean
            invert = false
        }
    }
}

lidar {
    # @recommended
    # @type = string
    # @description = "ROS topic for lidar data"
    topic = "/ouster/points"

    # @recommended
    # @type = enum[Velodyne,Ouster,Livox,Robosense,LivoxMid360]
    # @description = "LiDAR vendor type"
    vendor = Ouster

    # @recommended
    # @type = enum[Reliable,BestEffort,SystemDefault]
    qos.reliability = Reliable

    # @type = number
    # @description = "Weight for ICP correction in the IEKF"
    covariance = 0.001

    # @type = boolean
    # @description = "Use incoming Lidar messages as a forced sync trigger"
    as_sync = false
}

imu {
    # @recommended
    # @type = string
    # @description = "ROS topic for IMU data"
    topic = "/ouster/imu"

    # @recommended
    # @type = enum[Reliable,BestEffort,SystemDefault]
    qos.reliability = Reliable

    # @type = time
    # @description = "Initial setup time"
    initial_setup = 100ms

    # @type = enum[CreateOrientationOnly,FilterAccelerationAndCreateOrientation,FilterAccelerationOnly,Disable]
    ahrs_filtering.method = FilterAccelerationAndCreateOrientation

    # @type = number
    # @description = "Initial guess in m/s^2 for initialization"
    gravity = 9.81
}

output {
    terminal {
        # @recommended
        # @type = boolean
        # @description = "Show the interactive TUI"
        ui = true

        # @type = boolean
        debug = false

        # @type = number
        debug_fps = 1

        # @recommended
        # @type = enum[Trace,Debug,Info,Warn,Error]
        log_level = Warn
    }

    # @recommended
    # @type = string
    # @description = "Which frame to publish the TF messages to"
    global_frame = "lio_init"
}

parallelism {
    # @recommended
    # @type = number
    # @description = "Maximum number of CPU threads used for parallel point-cloud processing"
    num_threads = 3

    # @type = boolean
    # @description = "Force CPU-only processing"
    cpu_only = false
}
