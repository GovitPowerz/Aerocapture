use std::process;

use aerocapture::config;
use aerocapture::data;
use aerocapture::simulation;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: aerocapture <config.toml>");
        process::exit(1);
    }

    // TOML config file path as CLI argument
    let toml_path = &args[1];
    let content = match std::fs::read_to_string(toml_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Cannot read {}: {}", toml_path, e);
            process::exit(1);
        }
    };
    let (sim_config, toml_config) = match config::SimInput::from_toml(&content) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error parsing TOML config: {}", e);
            process::exit(1);
        }
    };

    let sim_data = if config::SimInput::is_consolidated(&toml_config) {
        // Consolidated mode: inline data + external files
        match data::SimData::from_toml(&toml_config, &sim_config) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Error loading inline data: {}", e);
                process::exit(1);
            }
        }
    } else {
        // Suffix mode: load all from external files
        match data::SimData::load(&sim_config) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Error loading data: {}", e);
                process::exit(1);
            }
        }
    };

    eprintln!(
        "Config: planet={:?}, nsims={}, guidance={:?}, reference={}, ref_bank={:.2}deg",
        sim_config.planet,
        sim_config.n_sims,
        sim_config.guidance_type,
        sim_config.reference_trajectory,
        sim_config.reference_bank_angle
    );
    eprintln!(
        "Data: base_dir='{}', output_dir='{}'",
        sim_config.base_dir, sim_config.output_dir
    );
    eprintln!(
        "RefTraj: {} points",
        sim_data.guidance.ref_trajectory.n_points
    );
    if sim_data.guidance.ref_trajectory.n_points > 0 {
        let rt = &sim_data.guidance.ref_trajectory;
        eprintln!(
            "  E[0]={:.3} MJ/kg, cos_bank[0]={:.6}, E[last]={:.3} MJ/kg",
            rt.energy[0] / 1e6,
            rt.cos_bank[0],
            rt.energy[rt.n_points - 1] / 1e6
        );
    }

    // Run simulation
    match simulation::runner::run(&sim_config, &sim_data) {
        Ok(()) => {}
        Err(e) => {
            eprintln!("Simulation error: {}", e);
            process::exit(1);
        }
    }
}
