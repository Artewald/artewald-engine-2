
/// WindowPixelPosition from top left corner of the window. Same as [winit::dpi::PhysicalPosition] as i32 mouse position.
pub struct WindowPixelPosition {
    pub x: i32,
    pub y: i32,
}

impl WindowPixelPosition {
    pub fn new(x: i32, y: i32) -> Self {
        Self {
            x,
            y,
        }
    }

    pub fn from_physical_position(physical_position: winit::dpi::PhysicalPosition<i32>) -> Self {
        Self {
            x: physical_position.x,
            y: physical_position.y,
        }
    }

    pub fn to_physical_position(&self) -> winit::dpi::PhysicalPosition<i32> {
        winit::dpi::PhysicalPosition {
            x: self.x,
            y: self.y,
        }
    }
}

/// The mouse scroll delta. Same as [winit::event::MouseScrollDelta::PixelDelta].
pub struct MouseScrollPixelDelta {
    pub x: f64,
    pub y: f64,
}

impl MouseScrollPixelDelta {
    pub fn new(x: f64, y: f64) -> Self {
        Self {
            x,
            y,
        }
    }

    pub fn from_winit(winit_scroll: winit::dpi::PhysicalPosition<f64>) -> Self {
        return Self {
            x: winit_scroll.x,
            y: winit_scroll.y,
        };
    }
}

pub enum KeyState {
    Pressed,
    Released,
    Held,
}

pub enum KeyType {
    Keyboard(KeyboardKeyCodes),
    MouseButton(MouseButtons),
}

/// The same as [winit::event::MouseButton]. This is a copy of the enum to avoid having to import winit as the end user.
pub enum MouseButtons {
    Left,
    Right,
    Middle,
    Other(u16),
}

impl MouseButtons {
    pub fn from_winit(winit_button: winit::event::MouseButton) -> Self {
        return match winit_button {
            winit::event::MouseButton::Left => Self::Left,
            winit::event::MouseButton::Right => Self::Right,
            winit::event::MouseButton::Middle => Self::Middle,
            winit::event::MouseButton::Other(x) => Self::Other(x),
        };
    }
}

/// The same as [winit::event::MouseScrollDelta]. This is a copy of the enum to avoid having to import winit as the end user.
pub enum MouseScrollDelta {
    LineDelta(f32, f32),
    PixelDelta(MouseScrollPixelDelta),
}

impl MouseScrollDelta {
    pub fn from_winit(winit_scroll: winit::event::MouseScrollDelta) -> Self {
        return match winit_scroll {
            winit::event::MouseScrollDelta::LineDelta(x, y) => Self::LineDelta(x, y),
            winit::event::MouseScrollDelta::PixelDelta(physical_position) => Self::PixelDelta(MouseScrollPixelDelta::from_winit(physical_position)),
        };
    }
}

/// The same as [winit::event::VirtualKeyCode]. This is a copy of the enum to avoid having to import winit as the end user.
pub enum KeyboardKeyCodes {
    /// The '1' key over the letters.
    Key1,
    /// The '2' key over the letters.
    Key2,
    /// The '3' key over the letters.
    Key3,
    /// The '4' key over the letters.
    Key4,
    /// The '5' key over the letters.
    Key5,
    /// The '6' key over the letters.
    Key6,
    /// The '7' key over the letters.
    Key7,
    /// The '8' key over the letters.
    Key8,
    /// The '9' key over the letters.
    Key9,
    /// The '0' key over the 'O' and 'P' keys.
    Key0,

    A,
    B,
    C,
    D,
    E,
    F,
    G,
    H,
    I,
    J,
    K,
    L,
    M,
    N,
    O,
    P,
    Q,
    R,
    S,
    T,
    U,
    V,
    W,
    X,
    Y,
    Z,

    /// The Escape key, next to F1.
    Escape,

    F1,
    F2,
    F3,
    F4,
    F5,
    F6,
    F7,
    F8,
    F9,
    F10,
    F11,
    F12,
    F13,
    F14,
    F15,
    F16,
    F17,
    F18,
    F19,
    F20,
    F21,
    F22,
    F23,
    F24,

    /// Print Screen/SysRq.
    Snapshot,
    /// Scroll Lock.
    Scroll,
    /// Pause/Break key, next to Scroll lock.
    Pause,

    /// `Insert`, next to Backspace.
    Insert,
    Home,
    Delete,
    End,
    PageDown,
    PageUp,

    Left,
    Up,
    Right,
    Down,

    /// The Backspace key, right over Enter.
    // TODO: rename
    Back,
    /// The Enter key.
    Return,
    /// The space bar.
    Space,

    /// The "Compose" key on Linux.
    Compose,

    Caret,

    Numlock,
    Numpad0,
    Numpad1,
    Numpad2,
    Numpad3,
    Numpad4,
    Numpad5,
    Numpad6,
    Numpad7,
    Numpad8,
    Numpad9,
    NumpadAdd,
    NumpadDivide,
    NumpadDecimal,
    NumpadComma,
    NumpadEnter,
    NumpadEquals,
    NumpadMultiply,
    NumpadSubtract,

    AbntC1,
    AbntC2,
    Apostrophe,
    Apps,
    Asterisk,
    At,
    Ax,
    Backslash,
    Calculator,
    Capital,
    Colon,
    Comma,
    Convert,
    Equals,
    Grave,
    Kana,
    Kanji,
    LAlt,
    LBracket,
    LControl,
    LShift,
    LWin,
    Mail,
    MediaSelect,
    MediaStop,
    Minus,
    Mute,
    MyComputer,
    // also called "Next"
    NavigateForward,
    // also called "Prior"
    NavigateBackward,
    NextTrack,
    NoConvert,
    OEM102,
    Period,
    PlayPause,
    Plus,
    Power,
    PrevTrack,
    RAlt,
    RBracket,
    RControl,
    RShift,
    RWin,
    Semicolon,
    Slash,
    Sleep,
    Stop,
    Sysrq,
    Tab,
    Underline,
    Unlabeled,
    VolumeDown,
    VolumeUp,
    Wake,
    WebBack,
    WebFavorites,
    WebForward,
    WebHome,
    WebRefresh,
    WebSearch,
    WebStop,
    Yen,
    Copy,
    Paste,
    Cut,
}

impl KeyboardKeyCodes {
    pub fn from_winit(winit_key: winit::event::VirtualKeyCode) -> Self {
        return match winit_key {
            winit::event::VirtualKeyCode::Key1 => Self::Key1,
            winit::event::VirtualKeyCode::Key2 => Self::Key2,
            winit::event::VirtualKeyCode::Key3 => Self::Key3,
            winit::event::VirtualKeyCode::Key4 => Self::Key4,
            winit::event::VirtualKeyCode::Key5 => Self::Key5,
            winit::event::VirtualKeyCode::Key6 => Self::Key6,
            winit::event::VirtualKeyCode::Key7 => Self::Key7,
            winit::event::VirtualKeyCode::Key8 => Self::Key8,
            winit::event::VirtualKeyCode::Key9 => Self::Key9,
            winit::event::VirtualKeyCode::Key0 => Self::Key0,
            winit::event::VirtualKeyCode::A => Self::A,
            winit::event::VirtualKeyCode::B => Self::B,
            winit::event::VirtualKeyCode::C => Self::C,
            winit::event::VirtualKeyCode::D => Self::D,
            winit::event::VirtualKeyCode::E => Self::E,
            winit::event::VirtualKeyCode::F => Self::F,
            winit::event::VirtualKeyCode::G => Self::G,
            winit::event::VirtualKeyCode::H => Self::H,
            winit::event::VirtualKeyCode::I => Self::I,
            winit::event::VirtualKeyCode::J => Self::J,
            winit::event::VirtualKeyCode::K => Self::K,
            winit::event::VirtualKeyCode::L => Self::L,
            winit::event::VirtualKeyCode::M => Self::M,
            winit::event::VirtualKeyCode::N => Self::N,
            winit::event::VirtualKeyCode::O => Self::O,
            winit::event::VirtualKeyCode::P => Self::P,
            winit::event::VirtualKeyCode::Q => Self::Q,
            winit::event::VirtualKeyCode::R => Self::R,
            winit::event::VirtualKeyCode::S => Self::S,
            winit::event::VirtualKeyCode::T => Self::T,
            winit::event::VirtualKeyCode::U => Self::U,
            winit::event::VirtualKeyCode::V => Self::V,
            winit::event::VirtualKeyCode::W => Self::W,
            winit::event::VirtualKeyCode::X => Self::X,
            winit::event::VirtualKeyCode::Y => Self::Y,
            winit::event::VirtualKeyCode::Z => Self::Z,
            winit::event::VirtualKeyCode::Escape => Self::Escape,
            winit::event::VirtualKeyCode::F1 => Self::F1,
            winit::event::VirtualKeyCode::F2 => Self::F2,
            winit::event::VirtualKeyCode::F3 => Self::F3,
            winit::event::VirtualKeyCode::F4 => Self::F4,
            winit::event::VirtualKeyCode::F5 => Self::F5,
            winit::event::VirtualKeyCode::F6 => Self::F6,
            winit::event::VirtualKeyCode::F7 => Self::F7,
            winit::event::VirtualKeyCode::F8 => Self::F8,
            winit::event::VirtualKeyCode::F9 => Self::F9,
            winit::event::VirtualKeyCode::F10 => Self::F10,
            winit::event::VirtualKeyCode::F11 => Self::F11,
            winit::event::VirtualKeyCode::F12 => Self::F12,
            winit::event::VirtualKeyCode::F13 => Self::F13,
            winit::event::VirtualKeyCode::F14 => Self::F14,
            winit::event::VirtualKeyCode::F15 => Self::F15,
            winit::event::VirtualKeyCode::F16 => Self::F16,
            winit::event::VirtualKeyCode::F17 => Self::F17,
            winit::event::VirtualKeyCode::F18 => Self::F18,
            winit::event::VirtualKeyCode::F19 => Self::F19,
            winit::event::VirtualKeyCode::F20 => Self::F20,
            winit::event::VirtualKeyCode::F21 => Self::F21,
            winit::event::VirtualKeyCode::F22 => Self::F22,
            winit::event::VirtualKeyCode::F23 => Self::F23,
            winit::event::VirtualKeyCode::F24 => Self::F24,
            winit::event::VirtualKeyCode::Snapshot => Self::Snapshot,
            winit::event::VirtualKeyCode::Scroll => Self::Scroll,
            winit::event::VirtualKeyCode::Pause => Self::Pause,
            winit::event::VirtualKeyCode::Insert => Self::Insert,
            winit::event::VirtualKeyCode::Home => Self::Home,
            winit::event::VirtualKeyCode::Delete => Self::Delete,
            winit::event::VirtualKeyCode::End => Self::End,
            winit::event::VirtualKeyCode::PageDown => Self::PageDown,
            winit::event::VirtualKeyCode::PageUp => Self::PageUp,
            winit::event::VirtualKeyCode::Left => Self::Left,
            winit::event::VirtualKeyCode::Up => Self::Up,
            winit::event::VirtualKeyCode::Right => Self::Right,
            winit::event::VirtualKeyCode::Down => Self::Down,
            winit::event::VirtualKeyCode::Back => Self::Back,
            winit::event::VirtualKeyCode::Return => Self::Return,
            winit::event::VirtualKeyCode::Space => Self::Space,
            winit::event::VirtualKeyCode::Compose => Self::Compose,
            winit::event::VirtualKeyCode::Caret => Self::Caret,
            winit::event::VirtualKeyCode::Numlock => Self::Numlock,
            winit::event::VirtualKeyCode::Numpad0 => Self::Numpad0,
            winit::event::VirtualKeyCode::Numpad1 => Self::Numpad1,
            winit::event::VirtualKeyCode::Numpad2 => Self::Numpad2,
            winit::event::VirtualKeyCode::Numpad3 => Self::Numpad3,
            winit::event::VirtualKeyCode::Numpad4 => Self::Numpad4,
            winit::event::VirtualKeyCode::Numpad5 => Self::Numpad5,
            winit::event::VirtualKeyCode::Numpad6 => Self::Numpad6,
            winit::event::VirtualKeyCode::Numpad7 => Self::Numpad7,
            winit::event::VirtualKeyCode::Numpad8 => Self::Numpad8,
            winit::event::VirtualKeyCode::Numpad9 => Self::Numpad9,
            winit::event::VirtualKeyCode::NumpadAdd => Self::NumpadAdd,
            winit::event::VirtualKeyCode::NumpadDivide => Self::NumpadDivide,
            winit::event::VirtualKeyCode::NumpadDecimal => Self::NumpadDecimal,
            winit::event::VirtualKeyCode::NumpadComma => Self::NumpadComma,
            winit::event::VirtualKeyCode::NumpadEnter => Self::NumpadEnter,
            winit::event::VirtualKeyCode::NumpadEquals => Self::NumpadEquals,
            winit::event::VirtualKeyCode::NumpadMultiply => Self::NumpadMultiply,
            winit::event::VirtualKeyCode::NumpadSubtract => Self::NumpadSubtract,
            winit::event::VirtualKeyCode::AbntC1 => Self::AbntC1,
            winit::event::VirtualKeyCode::AbntC2 => Self::AbntC2,
            winit::event::VirtualKeyCode::Apostrophe => Self::Apostrophe,
            winit::event::VirtualKeyCode::Apps => Self::Apps,
            winit::event::VirtualKeyCode::Asterisk => Self::Asterisk,
            winit::event::VirtualKeyCode::At => Self::At,
            winit::event::VirtualKeyCode::Ax => Self::Ax,
            winit::event::VirtualKeyCode::Backslash => Self::Backslash,
            winit::event::VirtualKeyCode::Calculator => Self::Calculator,
            winit::event::VirtualKeyCode::Capital => Self::Capital,
            winit::event::VirtualKeyCode::Colon => Self::Colon,
            winit::event::VirtualKeyCode::Comma => Self::Comma,
            winit::event::VirtualKeyCode::Convert => Self::Convert,
            winit::event::VirtualKeyCode::Equals => Self::Equals,
            winit::event::VirtualKeyCode::Grave => Self::Grave,
            winit::event::VirtualKeyCode::Kana => Self::Kana,
            winit::event::VirtualKeyCode::Kanji => Self::Kanji,
            winit::event::VirtualKeyCode::LAlt => Self::LAlt,
            winit::event::VirtualKeyCode::LBracket => Self::LBracket,
            winit::event::VirtualKeyCode::LControl => Self::LControl,
            winit::event::VirtualKeyCode::LShift => Self::LShift,
            winit::event::VirtualKeyCode::LWin => Self::LWin,
            winit::event::VirtualKeyCode::Mail => Self::Mail,
            winit::event::VirtualKeyCode::MediaSelect => Self::MediaSelect,
            winit::event::VirtualKeyCode::MediaStop => Self::MediaStop,
            winit::event::VirtualKeyCode::Minus => Self::Minus,
            winit::event::VirtualKeyCode::Mute => Self::Mute,
            winit::event::VirtualKeyCode::MyComputer => Self::MyComputer,
            winit::event::VirtualKeyCode::NavigateForward => Self::NavigateForward,
            winit::event::VirtualKeyCode::NavigateBackward => Self::NavigateBackward,
            winit::event::VirtualKeyCode::NextTrack => Self::NextTrack,
            winit::event::VirtualKeyCode::NoConvert => Self::NoConvert,
            winit::event::VirtualKeyCode::OEM102 => Self::OEM102,
            winit::event::VirtualKeyCode::Period => Self::Period,
            winit::event::VirtualKeyCode::PlayPause => Self::PlayPause,
            winit::event::VirtualKeyCode::Plus => Self::Plus,
            winit::event::VirtualKeyCode::Power => Self::Power,
            winit::event::VirtualKeyCode::PrevTrack => Self::PrevTrack,
            winit::event::VirtualKeyCode::RAlt => Self::RAlt,
            winit::event::VirtualKeyCode::RBracket => Self::RBracket,
            winit::event::VirtualKeyCode::RControl => Self::RControl,
            winit::event::VirtualKeyCode::RShift => Self::RShift,
            winit::event::VirtualKeyCode::RWin => Self::RWin,
            winit::event::VirtualKeyCode::Semicolon => Self::Semicolon,
            winit::event::VirtualKeyCode::Slash => Self::Slash,
            winit::event::VirtualKeyCode::Sleep => Self::Sleep,
            winit::event::VirtualKeyCode::Stop => Self::Stop,
            winit::event::VirtualKeyCode::Sysrq => Self::Sysrq,
            winit::event::VirtualKeyCode::Tab => Self::Tab,
            winit::event::VirtualKeyCode::Underline => Self::Underline,
            winit::event::VirtualKeyCode::Unlabeled => Self::Unlabeled,
            winit::event::VirtualKeyCode::VolumeDown => Self::VolumeDown,
            winit::event::VirtualKeyCode::VolumeUp => Self::VolumeUp,
            winit::event::VirtualKeyCode::Wake => Self::Wake,
            winit::event::VirtualKeyCode::WebBack => Self::WebBack,
            winit::event::VirtualKeyCode::WebFavorites => Self::WebFavorites,
            winit::event::VirtualKeyCode::WebForward => Self::WebForward,
            winit::event::VirtualKeyCode::WebHome => Self::WebHome,
            winit::event::VirtualKeyCode::WebRefresh => Self::WebRefresh,
            winit::event::VirtualKeyCode::WebSearch => Self::WebSearch,
            winit::event::VirtualKeyCode::WebStop => Self::WebStop,
            winit::event::VirtualKeyCode::Yen => Self::Yen,
            winit::event::VirtualKeyCode::Copy => Self::Copy,
            winit::event::VirtualKeyCode::Paste => Self::Paste,
            winit::event::VirtualKeyCode::Cut => Self::Cut,
        };
    }
}