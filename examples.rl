robot {
    sensor {
        lidar_range = 10m..50m
        lidar_range_feet = 30ft..150ft
        lidar_range_km = 0.01km..0.05km
        camera_fov = 60degrees
    }
    
    navigation {
        max_distance = 1light_year
        waypoint_precision = 1micron
        map_grid = 0.5m
    }
    
    wheel {
        radius = 5cm
        diameter_in = 4in
        diameter_imperial = 4inches
    }
}

time {
    update_rate = 10ms
    slow_rate = 1s
    period = 30min
    long_period = 1h
    daily_task = 1d
    yearly_maintenance = 1a
    sidereal_day = 1day_sidereal
    tropical_year = 1year_tropical
    pulse = 10shake
}

space {
    earth_sun = 1.5au
    star_distance = 4.37light_years
    space_station = 400km
    atom = 1angstrom
    quantum = 1bohr_radius
    nucleus = 10fermi
}
