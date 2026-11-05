"""Capture the first native access violation and x64 registers on Windows.

  python -m tools.blas_debug --out runs/verification/native_blas/debug.json -- \
      -m tools.blas_probe --worker --case gemm --size 1024 --out runs/scratch/blas

The child is terminated after recording the fault, without a WER dialog.
"""
import argparse
import ctypes as c
from ctypes import wintypes as w
import json
import os
from pathlib import Path
import struct
import subprocess
import sys


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("command", nargs=argparse.REMAINDER)
args = parser.parse_args()
command = args.command[1:] if args.command[:1] == ["--"] else args.command
if sys.platform != "win32":
    parser.error("requires Windows x64")
args.out.parent.mkdir(parents=True, exist_ok=True)
win = c.WinDLL("kernel32", use_last_error=True)
nt = c.WinDLL("ntdll")


class DebugEvent(c.Structure):
    _fields_ = [("code", w.DWORD), ("pid", w.DWORD), ("tid", w.DWORD),
                ("padding", w.DWORD), ("data", c.c_ubyte * 160)]


win.WaitForDebugEvent.argtypes = [c.POINTER(DebugEvent), w.DWORD]
win.ContinueDebugEvent.argtypes = [w.DWORD, w.DWORD, w.DWORD]
win.OpenThread.argtypes = [w.DWORD, w.BOOL, w.DWORD]
win.OpenThread.restype = w.HANDLE
win.GetThreadContext.argtypes = [w.HANDLE, c.c_void_p]
win.ReadProcessMemory.argtypes = [w.HANDLE, c.c_void_p, c.c_void_p, c.c_size_t, c.c_void_p]
win.TerminateProcess.argtypes = [w.HANDLE, w.UINT]
win.CloseHandle.argtypes = [w.HANDLE]
nt.NtQueryInformationThread.argtypes = [w.HANDLE, w.ULONG, c.c_void_p, w.ULONG, c.c_void_p]
child = subprocess.Popen([sys.executable, "-X", "faulthandler"] + command,
                         creationflags=0x08000002,
                         env=dict(os.environ, OPENBLAS_CORETYPE="Haswell",
                                  OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"))
event = DebugEvent()
report = dict(command=command, pid=child.pid)
while True:
    if not win.WaitForDebugEvent(c.byref(event), 60000):
        child.kill()
        raise TimeoutError("debug event wait failed: %s" % c.get_last_error())
    status = 0x00010002  # DBG_CONTINUE
    if event.code == 1:
        raw = bytes(event.data)
        code, flags, record, address, count = struct.unpack_from("<IIQQI", raw)
        if code == 0xC0000005 and "fault" not in report:
            access, target = struct.unpack_from("<QQ", raw, 32)
            thread = win.OpenThread(0x0048, False, event.tid)
            context = c.create_string_buffer(1248)
            context_address = (c.addressof(context) + 15) & ~15
            c.c_uint32.from_address(context_address + 48).value = 0x100003
            if not win.GetThreadContext(thread, context_address):
                child.kill()
                raise OSError(c.get_last_error(), "GetThreadContext")
            registers = dict(zip(
                ["rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
                 "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "rip"],
                struct.unpack("<17Q", c.string_at(context_address + 120, 136))))
            info = c.create_string_buffer(48)
            if nt.NtQueryInformationThread(thread, 0, info, len(info), None) != 0:
                child.kill()
                raise OSError("NtQueryInformationThread")
            teb = struct.unpack_from("<Q", info.raw, 8)[0]
            stack = c.create_string_buffer(16)
            if not win.ReadProcessMemory(int(child._handle), teb + 8, stack, 16, None):
                child.kill()
                raise OSError(c.get_last_error(), "ReadProcessMemory")
            stack_base, stack_limit = struct.unpack("<QQ", stack.raw)
            report["fault"] = dict(code=hex(code), instruction=hex(address),
                                    access=access, target=hex(target),
                                    stack_base=hex(stack_base), stack_limit=hex(stack_limit),
                                    registers={k: hex(v) for k, v in registers.items()})
            if access == 1 and 0 < registers["r12"] <= 512 and registers["rsi"] > registers["rsp"]:
                report["fault"].update(
                    buffer_start=hex(registers["rsp"] + 128),
                    bytes_written=registers["rsi"] - registers["rsp"] - 128,
                    initial_k=registers["r12"], remaining_k=registers["rax"])
            win.CloseHandle(thread)
            args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
            win.TerminateProcess(int(child._handle), code)
        elif code != 0x80000003:
            status = 0x80010001  # DBG_EXCEPTION_NOT_HANDLED
    elif event.code == 3 or event.code == 6:
        handle = struct.unpack_from("<Q", bytes(event.data))[0]
        if handle:
            win.CloseHandle(handle)
    elif event.code == 5:
        report["exit_code"] = struct.unpack_from("<I", bytes(event.data))[0]
        win.ContinueDebugEvent(event.pid, event.tid, status)
        break
    win.ContinueDebugEvent(event.pid, event.tid, status)
child.wait()
args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report), flush=True)
