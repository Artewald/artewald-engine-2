#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
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

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
/// The mouse scroll delta. Same as [winit::event::MouseScrollDelta::PixelDelta].
pub struct MouseScrollPixelDelta {
    pub x: f64,
    pub y: f64,
}

impl Into<MouseScrollPixelDelta> for winit::dpi::PhysicalPosition<f64> {
    fn into(self) -> MouseScrollPixelDelta {
        MouseScrollPixelDelta {
            x: self.x,
            y: self.y,
        }
    }
}

impl MouseScrollPixelDelta {
    pub fn new(x: f64, y: f64) -> Self {
        Self {
            x,
            y,
        }
    }
}

pub enum KeyState {
    Pressed,
    Released,
    Held,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum KeyType {
    Keyboard(KeyboardKeyCodes),
    MouseButton(MouseButtons),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
/// The same as [winit::event::MouseButton]. This is a copy of the enum to avoid having to import winit as the end user.
pub enum MouseButtons {
    Left,
    Right,
    Middle,
    Other(u16),
}

impl Into<MouseButtons> for winit::event::MouseButton {
    fn into(self) -> MouseButtons {
        match self {
            winit::event::MouseButton::Left => MouseButtons::Left,
            winit::event::MouseButton::Right => MouseButtons::Right,
            winit::event::MouseButton::Middle => MouseButtons::Middle,
            winit::event::MouseButton::Other(x) => MouseButtons::Other(x),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
/// The same as [winit::event::MouseScrollDelta]. This is a copy of the enum to avoid having to import winit as the end user.
pub enum MouseScrollDelta {
    LineDelta(f32, f32),
    PixelDelta(MouseScrollPixelDelta),
}

impl Into<MouseScrollDelta> for winit::event::MouseScrollDelta {
    fn into(self) -> MouseScrollDelta {
        match self {
            winit::event::MouseScrollDelta::LineDelta(x, y) => MouseScrollDelta::LineDelta(x, y),
            winit::event::MouseScrollDelta::PixelDelta(physical_position) => MouseScrollDelta::PixelDelta(physical_position.into()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
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

impl Into<KeyboardKeyCodes> for winit::event::VirtualKeyCode {
    fn into(self) -> KeyboardKeyCodes {
        match self {
            winit::event::VirtualKeyCode::Key1 => KeyboardKeyCodes::Key1,
            winit::event::VirtualKeyCode::Key2 => KeyboardKeyCodes::Key2,
            winit::event::VirtualKeyCode::Key3 => KeyboardKeyCodes::Key3,
            winit::event::VirtualKeyCode::Key4 => KeyboardKeyCodes::Key4,
            winit::event::VirtualKeyCode::Key5 => KeyboardKeyCodes::Key5,
            winit::event::VirtualKeyCode::Key6 => KeyboardKeyCodes::Key6,
            winit::event::VirtualKeyCode::Key7 => KeyboardKeyCodes::Key7,
            winit::event::VirtualKeyCode::Key8 => KeyboardKeyCodes::Key8,
            winit::event::VirtualKeyCode::Key9 => KeyboardKeyCodes::Key9,
            winit::event::VirtualKeyCode::Key0 => KeyboardKeyCodes::Key0,
            winit::event::VirtualKeyCode::A => KeyboardKeyCodes::A,
            winit::event::VirtualKeyCode::B => KeyboardKeyCodes::B,
            winit::event::VirtualKeyCode::C => KeyboardKeyCodes::C,
            winit::event::VirtualKeyCode::D => KeyboardKeyCodes::D,
            winit::event::VirtualKeyCode::E => KeyboardKeyCodes::E,
            winit::event::VirtualKeyCode::F => KeyboardKeyCodes::F,
            winit::event::VirtualKeyCode::G => KeyboardKeyCodes::G,
            winit::event::VirtualKeyCode::H => KeyboardKeyCodes::H,
            winit::event::VirtualKeyCode::I => KeyboardKeyCodes::I,
            winit::event::VirtualKeyCode::J => KeyboardKeyCodes::J,
            winit::event::VirtualKeyCode::K => KeyboardKeyCodes::K,
            winit::event::VirtualKeyCode::L => KeyboardKeyCodes::L,
            winit::event::VirtualKeyCode::M => KeyboardKeyCodes::M,
            winit::event::VirtualKeyCode::N => KeyboardKeyCodes::N,
            winit::event::VirtualKeyCode::O => KeyboardKeyCodes::O,
            winit::event::VirtualKeyCode::P => KeyboardKeyCodes::P,
            winit::event::VirtualKeyCode::Q => KeyboardKeyCodes::Q,
            winit::event::VirtualKeyCode::R => KeyboardKeyCodes::R,
            winit::event::VirtualKeyCode::S => KeyboardKeyCodes::S,
            winit::event::VirtualKeyCode::T => KeyboardKeyCodes::T,
            winit::event::VirtualKeyCode::U => KeyboardKeyCodes::U,
            winit::event::VirtualKeyCode::V => KeyboardKeyCodes::V,
            winit::event::VirtualKeyCode::W => KeyboardKeyCodes::W,
            winit::event::VirtualKeyCode::X => KeyboardKeyCodes::X,
            winit::event::VirtualKeyCode::Y => KeyboardKeyCodes::Y,
            winit::event::VirtualKeyCode::Z => KeyboardKeyCodes::Z,
            winit::event::VirtualKeyCode::Escape => KeyboardKeyCodes::Escape,
            winit::event::VirtualKeyCode::F1 => KeyboardKeyCodes::F1,
            winit::event::VirtualKeyCode::F2 => KeyboardKeyCodes::F2,
            winit::event::VirtualKeyCode::F3 => KeyboardKeyCodes::F3,
            winit::event::VirtualKeyCode::F4 => KeyboardKeyCodes::F4,
            winit::event::VirtualKeyCode::F5 => KeyboardKeyCodes::F5,
            winit::event::VirtualKeyCode::F6 => KeyboardKeyCodes::F6,
            winit::event::VirtualKeyCode::F7 => KeyboardKeyCodes::F7,
            winit::event::VirtualKeyCode::F8 => KeyboardKeyCodes::F8,
            winit::event::VirtualKeyCode::F9 => KeyboardKeyCodes::F9,
            winit::event::VirtualKeyCode::F10 => KeyboardKeyCodes::F10,
            winit::event::VirtualKeyCode::F11 => KeyboardKeyCodes::F11,
            winit::event::VirtualKeyCode::F12 => KeyboardKeyCodes::F12,
            winit::event::VirtualKeyCode::F13 => KeyboardKeyCodes::F13,
            winit::event::VirtualKeyCode::F14 => KeyboardKeyCodes::F14,
            winit::event::VirtualKeyCode::F15 => KeyboardKeyCodes::F15,
            winit::event::VirtualKeyCode::F16 => KeyboardKeyCodes::F16,
            winit::event::VirtualKeyCode::F17 => KeyboardKeyCodes::F17,
            winit::event::VirtualKeyCode::F18 => KeyboardKeyCodes::F18,
            winit::event::VirtualKeyCode::F19 => KeyboardKeyCodes::F19,
            winit::event::VirtualKeyCode::F20 => KeyboardKeyCodes::F20,
            winit::event::VirtualKeyCode::F21 => KeyboardKeyCodes::F21,
            winit::event::VirtualKeyCode::F22 => KeyboardKeyCodes::F22,
            winit::event::VirtualKeyCode::F23 => KeyboardKeyCodes::F23,
            winit::event::VirtualKeyCode::F24 => KeyboardKeyCodes::F24,
            winit::event::VirtualKeyCode::Snapshot => KeyboardKeyCodes::Snapshot,
            winit::event::VirtualKeyCode::Scroll => KeyboardKeyCodes::Scroll,
            winit::event::VirtualKeyCode::Pause => KeyboardKeyCodes::Pause,
            winit::event::VirtualKeyCode::Insert => KeyboardKeyCodes::Insert,
            winit::event::VirtualKeyCode::Home => KeyboardKeyCodes::Home,
            winit::event::VirtualKeyCode::Delete => KeyboardKeyCodes::Delete,
            winit::event::VirtualKeyCode::End => KeyboardKeyCodes::End,
            winit::event::VirtualKeyCode::PageDown => KeyboardKeyCodes::PageDown,
            winit::event::VirtualKeyCode::PageUp => KeyboardKeyCodes::PageUp,
            winit::event::VirtualKeyCode::Left => KeyboardKeyCodes::Left,
            winit::event::VirtualKeyCode::Up => KeyboardKeyCodes::Up,
            winit::event::VirtualKeyCode::Right => KeyboardKeyCodes::Right,
            winit::event::VirtualKeyCode::Down => KeyboardKeyCodes::Down,
            winit::event::VirtualKeyCode::Back => KeyboardKeyCodes::Back,
            winit::event::VirtualKeyCode::Return => KeyboardKeyCodes::Return,
            winit::event::VirtualKeyCode::Space => KeyboardKeyCodes::Space,
            winit::event::VirtualKeyCode::Compose => KeyboardKeyCodes::Compose,
            winit::event::VirtualKeyCode::Caret => KeyboardKeyCodes::Caret,
            winit::event::VirtualKeyCode::Numlock => KeyboardKeyCodes::Numlock,
            winit::event::VirtualKeyCode::Numpad0 => KeyboardKeyCodes::Numpad0,
            winit::event::VirtualKeyCode::Numpad1 => KeyboardKeyCodes::Numpad1,
            winit::event::VirtualKeyCode::Numpad2 => KeyboardKeyCodes::Numpad2,
            winit::event::VirtualKeyCode::Numpad3 => KeyboardKeyCodes::Numpad3,
            winit::event::VirtualKeyCode::Numpad4 => KeyboardKeyCodes::Numpad4,
            winit::event::VirtualKeyCode::Numpad5 => KeyboardKeyCodes::Numpad5,
            winit::event::VirtualKeyCode::Numpad6 => KeyboardKeyCodes::Numpad6,
            winit::event::VirtualKeyCode::Numpad7 => KeyboardKeyCodes::Numpad7,
            winit::event::VirtualKeyCode::Numpad8 => KeyboardKeyCodes::Numpad8,
            winit::event::VirtualKeyCode::Numpad9 => KeyboardKeyCodes::Numpad9,
            winit::event::VirtualKeyCode::NumpadAdd => KeyboardKeyCodes::NumpadAdd,
            winit::event::VirtualKeyCode::NumpadDivide => KeyboardKeyCodes::NumpadDivide,
            winit::event::VirtualKeyCode::NumpadDecimal => KeyboardKeyCodes::NumpadDecimal,
            winit::event::VirtualKeyCode::NumpadComma => KeyboardKeyCodes::NumpadComma,
            winit::event::VirtualKeyCode::NumpadEnter => KeyboardKeyCodes::NumpadEnter,
            winit::event::VirtualKeyCode::NumpadEquals => KeyboardKeyCodes::NumpadEquals,
            winit::event::VirtualKeyCode::NumpadMultiply => KeyboardKeyCodes::NumpadMultiply,
            winit::event::VirtualKeyCode::NumpadSubtract => KeyboardKeyCodes::NumpadSubtract,
            winit::event::VirtualKeyCode::AbntC1 => KeyboardKeyCodes::AbntC1,
            winit::event::VirtualKeyCode::AbntC2 => KeyboardKeyCodes::AbntC2,
            winit::event::VirtualKeyCode::Apostrophe => KeyboardKeyCodes::Apostrophe,
            winit::event::VirtualKeyCode::Apps => KeyboardKeyCodes::Apps,
            winit::event::VirtualKeyCode::Asterisk => KeyboardKeyCodes::Asterisk,
            winit::event::VirtualKeyCode::At => KeyboardKeyCodes::At,
            winit::event::VirtualKeyCode::Ax => KeyboardKeyCodes::Ax,
            winit::event::VirtualKeyCode::Backslash => KeyboardKeyCodes::Backslash,
            winit::event::VirtualKeyCode::Calculator => KeyboardKeyCodes::Calculator,
            winit::event::VirtualKeyCode::Capital => KeyboardKeyCodes::Capital,
            winit::event::VirtualKeyCode::Colon => KeyboardKeyCodes::Colon,
            winit::event::VirtualKeyCode::Comma => KeyboardKeyCodes::Comma,
            winit::event::VirtualKeyCode::Convert => KeyboardKeyCodes::Convert,
            winit::event::VirtualKeyCode::Equals => KeyboardKeyCodes::Equals,
            winit::event::VirtualKeyCode::Grave => KeyboardKeyCodes::Grave,
            winit::event::VirtualKeyCode::Kana => KeyboardKeyCodes::Kana,
            winit::event::VirtualKeyCode::Kanji => KeyboardKeyCodes::Kanji,
            winit::event::VirtualKeyCode::LAlt => KeyboardKeyCodes::LAlt,
            winit::event::VirtualKeyCode::LBracket => KeyboardKeyCodes::LBracket,
            winit::event::VirtualKeyCode::LControl => KeyboardKeyCodes::LControl,
            winit::event::VirtualKeyCode::LShift => KeyboardKeyCodes::LShift,
            winit::event::VirtualKeyCode::LWin => KeyboardKeyCodes::LWin,
            winit::event::VirtualKeyCode::Mail => KeyboardKeyCodes::Mail,
            winit::event::VirtualKeyCode::MediaSelect => KeyboardKeyCodes::MediaSelect,
            winit::event::VirtualKeyCode::MediaStop => KeyboardKeyCodes::MediaStop,
            winit::event::VirtualKeyCode::Minus => KeyboardKeyCodes::Minus,
            winit::event::VirtualKeyCode::Mute => KeyboardKeyCodes::Mute,
            winit::event::VirtualKeyCode::MyComputer => KeyboardKeyCodes::MyComputer,
            winit::event::VirtualKeyCode::NavigateForward => KeyboardKeyCodes::NavigateForward,
            winit::event::VirtualKeyCode::NavigateBackward => KeyboardKeyCodes::NavigateBackward,
            winit::event::VirtualKeyCode::NextTrack => KeyboardKeyCodes::NextTrack,
            winit::event::VirtualKeyCode::NoConvert => KeyboardKeyCodes::NoConvert,
            winit::event::VirtualKeyCode::OEM102 => KeyboardKeyCodes::OEM102,
            winit::event::VirtualKeyCode::Period => KeyboardKeyCodes::Period,
            winit::event::VirtualKeyCode::PlayPause => KeyboardKeyCodes::PlayPause,
            winit::event::VirtualKeyCode::Plus => KeyboardKeyCodes::Plus,
            winit::event::VirtualKeyCode::Power => KeyboardKeyCodes::Power,
            winit::event::VirtualKeyCode::PrevTrack => KeyboardKeyCodes::PrevTrack,
            winit::event::VirtualKeyCode::RAlt => KeyboardKeyCodes::RAlt,
            winit::event::VirtualKeyCode::RBracket => KeyboardKeyCodes::RBracket,
            winit::event::VirtualKeyCode::RControl => KeyboardKeyCodes::RControl,
            winit::event::VirtualKeyCode::RShift => KeyboardKeyCodes::RShift,
            winit::event::VirtualKeyCode::RWin => KeyboardKeyCodes::RWin,
            winit::event::VirtualKeyCode::Semicolon => KeyboardKeyCodes::Semicolon,
            winit::event::VirtualKeyCode::Slash => KeyboardKeyCodes::Slash,
            winit::event::VirtualKeyCode::Sleep => KeyboardKeyCodes::Sleep,
            winit::event::VirtualKeyCode::Stop => KeyboardKeyCodes::Stop,
            winit::event::VirtualKeyCode::Sysrq => KeyboardKeyCodes::Sysrq,
            winit::event::VirtualKeyCode::Tab => KeyboardKeyCodes::Tab,
            winit::event::VirtualKeyCode::Underline => KeyboardKeyCodes::Underline,
            winit::event::VirtualKeyCode::Unlabeled => KeyboardKeyCodes::Unlabeled,
            winit::event::VirtualKeyCode::VolumeDown => KeyboardKeyCodes::VolumeDown,
            winit::event::VirtualKeyCode::VolumeUp => KeyboardKeyCodes::VolumeUp,
            winit::event::VirtualKeyCode::Wake => KeyboardKeyCodes::Wake,
            winit::event::VirtualKeyCode::WebBack => KeyboardKeyCodes::WebBack,
            winit::event::VirtualKeyCode::WebFavorites => KeyboardKeyCodes::WebFavorites,
            winit::event::VirtualKeyCode::WebForward => KeyboardKeyCodes::WebForward,
            winit::event::VirtualKeyCode::WebHome => KeyboardKeyCodes::WebHome,
            winit::event::VirtualKeyCode::WebRefresh => KeyboardKeyCodes::WebRefresh,
            winit::event::VirtualKeyCode::WebSearch => KeyboardKeyCodes::WebSearch,
            winit::event::VirtualKeyCode::WebStop => KeyboardKeyCodes::WebStop,
            winit::event::VirtualKeyCode::Yen => KeyboardKeyCodes::Yen,
            winit::event::VirtualKeyCode::Copy => KeyboardKeyCodes::Copy,
            winit::event::VirtualKeyCode::Paste => KeyboardKeyCodes::Paste,
            winit::event::VirtualKeyCode::Cut => KeyboardKeyCodes::Cut,
        }
    }
}
