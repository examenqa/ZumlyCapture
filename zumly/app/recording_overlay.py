import ctypes
from ctypes import wintypes
import threading

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32
kernel32 = ctypes.windll.kernel32

WNDPROCTYPE = ctypes.WINFUNCTYPE(wintypes.LPARAM, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)

WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOPMOST = 0x00000008
WS_EX_TOOLWINDOW = 0x00000080
WS_POPUP = 0x80000000

WDA_EXCLUDEFROMCAPTURE = 0x00000011

WM_PAINT = 0x000F
WM_DESTROY = 0x0002
WM_QUIT = 0x0012
SW_SHOWNOACTIVATE = 4
HWND_TOPMOST = -1
SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOACTIVATE = 0x0010
BASE_DPI = 96.0
BI_RGB = 0
DIB_RGB_COLORS = 0
ULW_ALPHA = 0x00000002
AC_SRC_OVER = 0x00
AC_SRC_ALPHA = 0x01
ANTIALIAS_GRID = 8

# Hallmark · component: recording status badge · genre: modern-minimal · theme: Cobalt
# Hallmark · pre-emit critique: P5 H5 E5 S5 R5 V4
BASE_WIDTH = 24
BASE_HEIGHT = 24
TOP_OFFSET = 16
OUTER_RADIUS = 10.5
RING_RADIUS = 8.75
DOT_RADIUS = 5.0

# COLORREF values are BGR. The cool neutral edge keeps the near-white ring
# visible against both light and dark content without growing the badge.
EDGE_COLOR = 0x002A170F       # RGB(15, 23, 42) / #0F172A
RING_COLOR = 0x00FCFAF8       # RGB(248, 250, 252) / #F8FAFC
RECORDING_COLOR = 0x004444EF  # RGB(239, 68, 68) / #EF4444
PAUSED_COLOR = 0x000B9EF5     # RGB(245, 158, 11) / #F59E0B
EDGE_ALPHA = 216


class BLENDFUNCTION(ctypes.Structure):
    _fields_ = [
        ("BlendOp", wintypes.BYTE),
        ("BlendFlags", wintypes.BYTE),
        ("SourceConstantAlpha", wintypes.BYTE),
        ("AlphaFormat", wintypes.BYTE),
    ]


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [
        ("bmiHeader", BITMAPINFOHEADER),
        ("bmiColors", wintypes.DWORD * 3),
    ]

class WNDCLASSEX(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.UINT),
                ("style", wintypes.UINT),
                ("lpfnWndProc", WNDPROCTYPE),
                ("cbClsExtra", ctypes.c_int),
                ("cbWndExtra", ctypes.c_int),
                ("hInstance", wintypes.HINSTANCE),
                ("hIcon", wintypes.HICON),
                ("hCursor", wintypes.HANDLE),
                ("hbrBackground", wintypes.HBRUSH),
                ("lpszMenuName", wintypes.LPCWSTR),
                ("lpszClassName", wintypes.LPCWSTR),
                ("hIconSm", wintypes.HICON)]

class PAINTSTRUCT(ctypes.Structure):
    _fields_ = [("hdc", wintypes.HDC),
                ("fErase", wintypes.BOOL),
                ("rcPaint", wintypes.RECT),
                ("fRestore", wintypes.BOOL),
                ("fIncUpdate", wintypes.BOOL),
                ("rgbReserved", ctypes.c_byte * 32)]

# Define argtypes for safety on 64-bit platforms
user32.CreateWindowExW.argtypes = [
    wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR,
    wintypes.DWORD, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, wintypes.HWND,
    wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID
]
user32.CreateWindowExW.restype = wintypes.HWND

user32.RegisterClassExW.argtypes = [ctypes.POINTER(WNDCLASSEX)]
user32.RegisterClassExW.restype = wintypes.ATOM

user32.DefWindowProcW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
user32.DefWindowProcW.restype = wintypes.LPARAM

user32.PostThreadMessageW.argtypes = [wintypes.DWORD, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
user32.PostThreadMessageW.restype = wintypes.BOOL
if hasattr(user32, "GetDpiForSystem"):
    user32.GetDpiForSystem.argtypes = []
    user32.GetDpiForSystem.restype = wintypes.UINT
user32.BeginPaint.argtypes = [wintypes.HWND, ctypes.POINTER(PAINTSTRUCT)]
user32.BeginPaint.restype = wintypes.HDC
user32.EndPaint.argtypes = [wintypes.HWND, ctypes.POINTER(PAINTSTRUCT)]
user32.EndPaint.restype = wintypes.BOOL
user32.SetWindowPos.argtypes = [
    wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, wintypes.UINT
]
user32.SetWindowPos.restype = wintypes.BOOL
user32.InvalidateRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT), wintypes.BOOL]
user32.InvalidateRect.restype = wintypes.BOOL
user32.UnregisterClassW.argtypes = [wintypes.LPCWSTR, wintypes.HINSTANCE]
user32.UnregisterClassW.restype = wintypes.BOOL
user32.GetDC.argtypes = [wintypes.HWND]
user32.GetDC.restype = wintypes.HDC
user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
user32.ReleaseDC.restype = ctypes.c_int
user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
user32.ClientToScreen.restype = wintypes.BOOL
user32.UpdateLayeredWindow.argtypes = [
    wintypes.HWND,
    wintypes.HDC,
    ctypes.POINTER(wintypes.POINT),
    ctypes.POINTER(wintypes.SIZE),
    wintypes.HDC,
    ctypes.POINTER(wintypes.POINT),
    wintypes.COLORREF,
    ctypes.POINTER(BLENDFUNCTION),
    wintypes.DWORD,
]
user32.UpdateLayeredWindow.restype = wintypes.BOOL


# Define GDI argtypes
gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HANDLE]
gdi32.SelectObject.restype = wintypes.HANDLE
gdi32.DeleteObject.argtypes = [wintypes.HANDLE]
gdi32.DeleteObject.restype = wintypes.BOOL
gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
gdi32.CreateCompatibleDC.restype = wintypes.HDC
gdi32.DeleteDC.argtypes = [wintypes.HDC]
gdi32.DeleteDC.restype = wintypes.BOOL
gdi32.CreateDIBSection.argtypes = [
    wintypes.HDC,
    ctypes.POINTER(BITMAPINFO),
    wintypes.UINT,
    ctypes.POINTER(ctypes.c_void_p),
    wintypes.HANDLE,
    wintypes.DWORD,
]
gdi32.CreateDIBSection.restype = wintypes.HBITMAP


def _colorref_rgb(color: int) -> tuple[int, int, int]:
    """Convert a Win32 COLORREF into an RGB tuple."""
    return color & 0xFF, (color >> 8) & 0xFF, (color >> 16) & 0xFF


def _render_indicator_pixels(width: int, height: int, paused: bool) -> bytes:
    """Render a smooth premultiplied-BGRA recording indicator."""
    scale = min(width, height) / BASE_WIDTH
    center_x = width / 2.0
    center_y = height / 2.0
    outer_radius = OUTER_RADIUS * scale
    ring_radius = RING_RADIUS * scale
    dot_radius = DOT_RADIUS * scale
    dot_color = PAUSED_COLOR if paused else RECORDING_COLOR
    layers = (
        (dot_radius * dot_radius, _colorref_rgb(dot_color), 255),
        (ring_radius * ring_radius, _colorref_rgb(RING_COLOR), 255),
        (outer_radius * outer_radius, _colorref_rgb(EDGE_COLOR), EDGE_ALPHA),
    )
    sample_count = ANTIALIAS_GRID * ANTIALIAS_GRID
    pixels = bytearray(width * height * 4)

    for y in range(height):
        for x in range(width):
            alpha_sum = red_sum = green_sum = blue_sum = 0
            for sample_y in range(ANTIALIAS_GRID):
                point_y = y + (sample_y + 0.5) / ANTIALIAS_GRID
                dy = point_y - center_y
                for sample_x in range(ANTIALIAS_GRID):
                    point_x = x + (sample_x + 0.5) / ANTIALIAS_GRID
                    dx = point_x - center_x
                    distance_squared = dx * dx + dy * dy
                    for radius_squared, (red, green, blue), alpha in layers:
                        if distance_squared <= radius_squared:
                            alpha_sum += alpha
                            red_sum += red * alpha
                            green_sum += green * alpha
                            blue_sum += blue * alpha
                            break

            offset = (y * width + x) * 4
            pixels[offset] = round(blue_sum / (sample_count * 255))
            pixels[offset + 1] = round(green_sum / (sample_count * 255))
            pixels[offset + 2] = round(red_sum / (sample_count * 255))
            pixels[offset + 3] = round(alpha_sum / sample_count)

    return bytes(pixels)


def _get_system_scale_factor() -> float:
    """Return the active system DPI scale relative to the 96-DPI baseline."""
    try:
        dpi = float(user32.GetDpiForSystem())
    except Exception:
        dpi = BASE_DPI
    if dpi <= 0:
        dpi = BASE_DPI
    return dpi / BASE_DPI


class RecordingOverlay:
    """A compact Win32 recording-status dot excluded from capture."""
    
    def __init__(self, monitor_rect: dict):
        self.monitor_rect = monitor_rect
        self.scale_factor = _get_system_scale_factor()
        self.width = self._px(BASE_WIDTH)
        self.height = self._px(BASE_HEIGHT)
        self.hwnd = None
        self.thread_id = None
        self._class_name = ""
        self._ready = threading.Event()
        self._state_lock = threading.Lock()
        self._paused = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        # Prevent garbage collection of the wndproc callback
        self._wndproc_c = WNDPROCTYPE(self._wndproc)

    def start(self):
        self._thread.start()
        self._ready.wait(timeout=1.0)

    def stop(self):
        if self.thread_id:
            user32.PostThreadMessageW(self.thread_id, WM_QUIT, 0, 0)
            self._thread.join(timeout=2.0)

    def set_paused(self, paused: bool) -> None:
        """Switch the lightweight GDI indicator between recording states."""
        with self._state_lock:
            self._paused = bool(paused)
            hwnd = self.hwnd
        if hwnd:
            user32.InvalidateRect(hwnd, None, True)

    def _px(self, value: float) -> int:
        return max(1, int(round(value * self.scale_factor)))

    def _present_indicator(self, hwnd, paused: bool) -> None:
        """Present the antialiased badge through a per-pixel-alpha layer."""
        screen_dc = user32.GetDC(0)
        memory_dc = gdi32.CreateCompatibleDC(screen_dc)
        bitmap = None
        old_bitmap = None
        try:
            bitmap_info = BITMAPINFO()
            bitmap_info.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
            bitmap_info.bmiHeader.biWidth = self.width
            bitmap_info.bmiHeader.biHeight = -self.height
            bitmap_info.bmiHeader.biPlanes = 1
            bitmap_info.bmiHeader.biBitCount = 32
            bitmap_info.bmiHeader.biCompression = BI_RGB
            bits = ctypes.c_void_p()
            bitmap = gdi32.CreateDIBSection(
                screen_dc,
                ctypes.byref(bitmap_info),
                DIB_RGB_COLORS,
                ctypes.byref(bits),
                None,
                0,
            )
            if not bitmap or not bits.value:
                return
            old_bitmap = gdi32.SelectObject(memory_dc, bitmap)
            pixel_data = _render_indicator_pixels(self.width, self.height, paused)
            ctypes.memmove(bits, pixel_data, len(pixel_data))

            destination = wintypes.POINT()
            user32.ClientToScreen(hwnd, ctypes.byref(destination))
            size = wintypes.SIZE(self.width, self.height)
            source = wintypes.POINT(0, 0)
            blend = BLENDFUNCTION(AC_SRC_OVER, 0, 255, AC_SRC_ALPHA)
            user32.UpdateLayeredWindow(
                hwnd,
                screen_dc,
                ctypes.byref(destination),
                ctypes.byref(size),
                memory_dc,
                ctypes.byref(source),
                0,
                ctypes.byref(blend),
                ULW_ALPHA,
            )
        finally:
            if old_bitmap:
                gdi32.SelectObject(memory_dc, old_bitmap)
            if bitmap:
                gdi32.DeleteObject(bitmap)
            if memory_dc:
                gdi32.DeleteDC(memory_dc)
            if screen_dc:
                user32.ReleaseDC(0, screen_dc)

    def _wndproc(self, hwnd, msg, wparam, lparam):
        if msg == WM_PAINT:
            with self._state_lock:
                paused = self._paused
            ps = PAINTSTRUCT()
            user32.BeginPaint(hwnd, ctypes.byref(ps))
            user32.EndPaint(hwnd, ctypes.byref(ps))
            self._present_indicator(hwnd, paused)
            return 0
        elif msg == WM_DESTROY:
            user32.PostQuitMessage(0)
            return 0
        
        return user32.DefWindowProcW(hwnd, msg, wparam, lparam)

    def _run(self):
        self.thread_id = kernel32.GetCurrentThreadId()

        class_name = f"ZumlyRecordingOverlay_{kernel32.GetCurrentProcessId()}_{self.thread_id}"
        self._class_name = class_name
        wndclass = WNDCLASSEX()
        wndclass.cbSize = ctypes.sizeof(WNDCLASSEX)
        wndclass.lpfnWndProc = self._wndproc_c
        wndclass.hInstance = kernel32.GetModuleHandleW(None)
        wndclass.lpszClassName = class_name
        
        wndclass.hbrBackground = None

        user32.RegisterClassExW(ctypes.byref(wndclass))

        # Position at the top-center of the recording monitor
        width = self.width
        height = self.height
        x = self.monitor_rect.get("left", 0) + (self.monitor_rect.get("width", 1920) - width) // 2
        y = self.monitor_rect.get("top", 0) + self._px(TOP_OFFSET)

        self.hwnd = user32.CreateWindowExW(
            WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_TOOLWINDOW,
            class_name,
            "ZumlyOverlay",
            WS_POPUP,
            x, y, width, height,
            0, 0, wndclass.hInstance, 0
        )

        if not self.hwnd:
            self._ready.set()
            return

        user32.SetWindowDisplayAffinity(self.hwnd, WDA_EXCLUDEFROMCAPTURE)

        user32.ShowWindow(self.hwnd, SW_SHOWNOACTIVATE)
        user32.SetWindowPos(
            self.hwnd,
            HWND_TOPMOST,
            0, 0, 0, 0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE,
        )
        user32.InvalidateRect(self.hwnd, None, True)
        user32.UpdateWindow(self.hwnd)
        self._ready.set()

        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), 0, 0, 0) > 0:
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))

        if self.hwnd:
            user32.DestroyWindow(self.hwnd)
            self.hwnd = None
        if self._class_name:
            user32.UnregisterClassW(self._class_name, wndclass.hInstance)
