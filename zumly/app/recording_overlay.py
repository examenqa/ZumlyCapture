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

LWA_COLORKEY = 1
WDA_EXCLUDEFROMCAPTURE = 0x00000011

WM_PAINT = 0x000F
WM_DESTROY = 0x0002
WM_QUIT = 0x0012
SW_SHOWNOACTIVATE = 4
HWND_TOPMOST = -1
SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOACTIVATE = 0x0010
NULL_PEN = 8
BASE_DPI = 96.0

# Hallmark · component: recording status badge · genre: modern-minimal · theme: Cobalt
# Pre-emit critique: P5 H5 E5 S5 R5 V4
BASE_WIDTH = 24
BASE_HEIGHT = 24
TOP_OFFSET = 16
OUTER_CIRCLE = (1, 1, 23, 23)
RING_CIRCLE = (3, 3, 21, 21)
DOT_CIRCLE = (7, 7, 17, 17)

# COLORREF values are BGR. The cool neutral edge keeps the near-white ring
# visible against both light and dark content without growing the badge.
EDGE_COLOR = 0x002A170F       # RGB(15, 23, 42) / #0F172A
RING_COLOR = 0x00FCFAF8       # RGB(248, 250, 252) / #F8FAFC
RECORDING_COLOR = 0x004444EF  # RGB(239, 68, 68) / #EF4444
PAUSED_COLOR = 0x000B9EF5     # RGB(245, 158, 11) / #F59E0B

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


# Define GDI argtypes
gdi32.GetStockObject.argtypes = [ctypes.c_int]
gdi32.GetStockObject.restype = wintypes.HANDLE
gdi32.CreateSolidBrush.argtypes = [wintypes.COLORREF]
gdi32.CreateSolidBrush.restype = wintypes.HBRUSH
gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HANDLE]
gdi32.SelectObject.restype = wintypes.HANDLE
gdi32.DeleteObject.argtypes = [wintypes.HANDLE]
gdi32.DeleteObject.restype = wintypes.BOOL
gdi32.Ellipse.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]


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

    def _draw_circle(self, hdc, bounds: tuple[int, int, int, int], color: int) -> None:
        brush = gdi32.CreateSolidBrush(color)
        old_brush = gdi32.SelectObject(hdc, brush)
        gdi32.Ellipse(hdc, *(self._px(value) for value in bounds))
        gdi32.SelectObject(hdc, old_brush)
        gdi32.DeleteObject(brush)

    def _wndproc(self, hwnd, msg, wparam, lparam):
        if msg == WM_PAINT:
            with self._state_lock:
                paused = self._paused
            ps = PAINTSTRUCT()
            hdc = user32.BeginPaint(hwnd, ctypes.byref(ps))

            # Remove black borders globally
            hpen_null = gdi32.GetStockObject(NULL_PEN)
            old_pen = gdi32.SelectObject(hdc, hpen_null)

            # Three concentric circles remain legible on any captured content.
            self._draw_circle(hdc, OUTER_CIRCLE, EDGE_COLOR)
            self._draw_circle(hdc, RING_CIRCLE, RING_COLOR)
            self._draw_circle(
                hdc,
                DOT_CIRCLE,
                PAUSED_COLOR if paused else RECORDING_COLOR,
            )
            gdi32.SelectObject(hdc, old_pen)

            user32.EndPaint(hwnd, ctypes.byref(ps))
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
        
        # Transparent colorkey background (Magenta)
        magenta = 0x00FF00FF
        hbrush = gdi32.CreateSolidBrush(magenta)
        wndclass.hbrBackground = hbrush

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

        user32.SetLayeredWindowAttributes(self.hwnd, magenta, 0, LWA_COLORKEY)
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
        gdi32.DeleteObject(hbrush)
