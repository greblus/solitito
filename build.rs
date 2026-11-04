fn main() {
    slint_build::compile("ui/appwindow.slint").unwrap();
    windows_icon();
}

/// Embeds `src/assets/app-icon.ico` in solitito.exe. Windows builds only.
#[cfg(windows)]
fn windows_icon() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        winresource::WindowsResource::new()
            .set_icon("src/assets/app-icon.ico")
            .compile()
            .expect("the Windows resource with the icon could not be built");
    }
}

#[cfg(not(windows))]
fn windows_icon() {}
