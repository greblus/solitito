//! OpenGL or software rendering of the window.
//!
//! Slint falls back to software only when the OpenGL renderer cannot be
//! created; when OpenGL fails while opening the window, `run` returns an
//! error. Then `software_renderer` is saved and the app restarts.
//! `SLINT_BACKEND`, when set, overrides all of this.

use std::env;
use std::process::Command;

/// `SLINT_BACKEND` is set.
fn chosen_by_hand() -> bool {
    env::var_os("SLINT_BACKEND").is_some()
}

/// Selects the software renderer if `software`. Call before the window is built.
pub fn select(software: bool) {
    if !software || chosen_by_hand() {
        return;
    }
    let selected = slint::BackendSelector::new()
        .backend_name("winit".into())
        .renderer_name("software".into())
        .select();
    match selected {
        Ok(()) => println!("🖥  Drawing in software"),
        Err(e) => eprintln!("The software renderer could not be selected: {e}"),
    }
}

/// Whether to restart in software after the window failed to open.
pub fn worth_retrying(software: bool) -> bool {
    !software && !chosen_by_hand()
}

/// Runs the program again with the same arguments and exits with its code.
pub fn start_again() -> ! {
    let code = env::current_exe()
        .and_then(|exe| Command::new(exe).args(env::args_os().skip(1)).status())
        .map(|status| status.code().unwrap_or(1));
    match code {
        Ok(code) => std::process::exit(code),
        Err(e) => {
            eprintln!("Could not start again in software: {e}");
            std::process::exit(1)
        }
    }
}
