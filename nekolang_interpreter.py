#!/usr/bin/env python3
# NekoLang - tiny interpreted language with UI, HTTP, imports and meow pkg manager

import re, math, subprocess, sys, os, io, json, pathlib, random
import tkinter as tk
from typing import List
from functools import partial
try:
    import requests
except ImportError:
    requests = None  # net & meow will error with clear message

# ----------------------------- Config -----------------------------
REGISTRY_URL = "https://raw.githubusercontent.com/comet-CS/nekolang-reg/refs/heads/main/packages.txt"

# resolve important dirs
INTERP_PATH = pathlib.Path(__file__).resolve()
INTERP_DIR  = INTERP_PATH.parent
LOCAL_LIBS  = (INTERP_DIR / "libs")             # next to interpreter
HOME_LIBS   = (pathlib.Path.home() / ".nekolang" / "libs")  # global per-user
CWD_LIBS    = (pathlib.Path.cwd() / "libs")     # project-local

for d in (LOCAL_LIBS, HOME_LIBS):
    d.mkdir(parents=True, exist_ok=True)

# ----------------------------- Tokenizer -----------------------------
TOKEN_SPEC = [
    ("COMMENT",  r"//[^\n]*"),
    ("STRING",   r'"([^"\\]|\\.)*"'),
    ("NUMBER",   r"\d+(\.\d+)?"),
    ("ID",       r"[A-Za-z_][A-Za-z0-9_]*"),
    ("OP",       r"==|!=|<=|>=|&&|\|\||[+\-*/<>.=:\[\]]"),
    ("ASSIGN",   r"="),   # kept for compatibility
    ("LPAREN",   r"\("),
    ("RPAREN",   r"\)"),
    ("LBRACE",   r"\{"),
    ("RBRACE",   r"\}"),
    ("COMMA",    r","),
    ("SEMICOL",  r";"),
    ("NEWLINE",  r"\n"),
    ("SKIP",     r"[ \t\r]+"),
]
MASTER_RE = re.compile("|".join("(?P<%s>%s)" % pair for pair in TOKEN_SPEC))

class Token:
    def __init__(self, t, v):
        self.type, self.value = t, v
    def __repr__(self):
        return f"Token({self.type},{self.value})"

def unescape_string(s: str) -> str:
    inner = s[1:-1]
    return bytes(inner, "utf-8").decode("unicode_escape")

def tokenize(code: str) -> List[Token]:
    tokens: List[Token] = []
    for mo in MASTER_RE.finditer(code):
        kind = mo.lastgroup
        value = mo.group()
        if kind == "NUMBER":
            value = float(value) if "." in value else int(value)
            tokens.append(Token("NUMBER", value))
        elif kind == "STRING":
            tokens.append(Token("STRING", unescape_string(value)))
        elif kind == "ID":
            # normalize boolean/not/and/or as IDs with same text; parser handles
            tokens.append(Token("ID", value))
        elif kind == "OP":
            tokens.append(Token("OP", value))
        elif kind in ("LPAREN","RPAREN","LBRACE","RBRACE","COMMA","SEMICOL","ASSIGN"):
            tokens.append(Token(kind, value))
        elif kind == "NEWLINE":
            tokens.append(Token("NEWLINE", value))
        elif kind in ("SKIP", "COMMENT"):
            continue
        else:
            raise SyntaxError(f"Unknown token {value}")
    tokens.append(Token("EOF",""))
    return tokens

# ----------------------------- Parser -----------------------------
class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens; self.pos = 0

    def cur(self) -> Token: return self.tokens[self.pos]
    def peek(self, n=1): return self.tokens[self.pos+n] if self.pos+n < len(self.tokens) else Token("EOF","")
    def eat(self, ttype=None, value=None) -> Token:
        tok = self.cur()
        if ttype and tok.type != ttype:
            raise SyntaxError(f"Expected {ttype} but got {tok.type} ({tok.value})")
        if value and tok.value != value:
            raise SyntaxError(f"Expected {value} but got {tok.value}")
        self.pos += 1
        return tok

    def parse(self):
        stmts = []
        while self.cur().type != "EOF":
            if self.cur().type == "NEWLINE":
                self.eat("NEWLINE"); continue
            stmts.append(self.parse_stmt())
        return ("BLOCK", stmts)

    def parse_stmt(self):
        tok = self.cur()
        if tok.type == "ID" and tok.value == "let":
            return self.parse_let()
        if tok.type == "ID" and tok.value == "print":
            return self.parse_print()
        if tok.type == "ID" and tok.value == "fn":
            return self.parse_fn()
        if tok.type == "ID" and tok.value == "return":
            return self.parse_return()
        if tok.type == "ID" and tok.value == "if":
            return self.parse_if()
        if tok.type == "ID" and tok.value == "while":
            return self.parse_while()
        if tok.type == "ID" and tok.value == "import":
            return self.parse_import()
        # expression statement
        expr = self.parse_expr()
        if self.cur().type == "SEMICOL":
            self.eat("SEMICOL")
        return ("EXPR_STMT", expr)

    def parse_import(self):
        self.eat("ID")  # import
        # import accepts string literal or identifier (name)
        if self.cur().type == "STRING":
            mod = self.eat("STRING").value
        else:
            mod = self.eat("ID").value
        if self.cur().type == "SEMICOL":
            self.eat("SEMICOL")
        return ("IMPORT", mod)

    def parse_let(self):
        self.eat("ID")  # let
        name = self.eat("ID").value
        self.eat("ASSIGN")
        expr = self.parse_expr()
        if self.cur().type == "SEMICOL": self.eat("SEMICOL")
        return ("LET", name, expr)

    def parse_print(self):
        self.eat("ID")
        expr = self.parse_expr()
        if self.cur().type == "SEMICOL": self.eat("SEMICOL")
        return ("PRINT", expr)

    def parse_return(self):
        self.eat("ID")
        expr = self.parse_expr()
        if self.cur().type == "SEMICOL": self.eat("SEMICOL")
        return ("RETURN", expr)

    def parse_fn(self):
        self.eat("ID")
        name = self.eat("ID").value
        self.eat("LPAREN")
        args = []
        if self.cur().type != "RPAREN":
            args.append(self.eat("ID").value)
            while self.cur().type == "COMMA":
                self.eat("COMMA")
                args.append(self.eat("ID").value)
        self.eat("RPAREN")
        body = self.parse_block()
        return ("FNDEF", name, args, body)

    def parse_if(self):
        self.eat("ID")
        cond = self.parse_expr()
        then_block = self.parse_block()
        else_block = None
        if self.cur().type == "ID" and self.cur().value == "else":
            self.eat("ID")
            else_block = self.parse_block()
        return ("IF", cond, then_block, else_block)

    def parse_while(self):
        self.eat("ID")
        cond = self.parse_expr()
        body = self.parse_block()
        return ("WHILE", cond, body)

    def parse_block(self):
        if self.cur().type == "LBRACE":
            self.eat("LBRACE")
            stmts = []
            while self.cur().type != "RBRACE":
                if self.cur().type == "NEWLINE":
                    self.eat("NEWLINE"); continue
                stmts.append(self.parse_stmt())
            self.eat("RBRACE")
            return ("BLOCK", stmts)
        else:
            stmt = self.parse_stmt()
            return ("BLOCK", [stmt])

    # -------- expressions with precedence: or -> and -> not -> comparison -> add -> mul -> unary -> primary
    def parse_expr(self): return self.parse_or()

    def parse_or(self):
        node = self.parse_and()
        while self.cur().type == "ID" and self.cur().value == "or" or (self.cur().type=="OP" and self.cur().value=="||"):
            if self.cur().type == "ID": self.eat("ID")
            else: self.eat("OP","||")
            right = self.parse_and()
            node = ("LOR", node, right)
        return node

    def parse_and(self):
        node = self.parse_not()
        while self.cur().type == "ID" and self.cur().value == "and" or (self.cur().type=="OP" and self.cur().value=="&&"):
            if self.cur().type == "ID": self.eat("ID")
            else: self.eat("OP","&&")
            right = self.parse_not()
            node = ("LAND", node, right)
        return node

    def parse_not(self):
        if self.cur().type == "ID" and self.cur().value == "not":
            self.eat("ID")
            node = self.parse_not()
            return ("LNOT", node)
        return self.parse_comparison()

    def parse_comparison(self):
        node = self.parse_add()
        while self.cur().type == "OP" and self.cur().value in ("==","!=", "<",">","<=",">="):
            op = self.eat("OP").value
            right = self.parse_add()
            node = ("BINOP", op, node, right)
        return node

    def parse_add(self):
        node = self.parse_mul()
        while self.cur().type == "OP" and self.cur().value in ("+","-"):
            op = self.eat("OP").value
            right = self.parse_mul()
            node = ("BINOP", op, node, right)
        return node

    def parse_mul(self):
        node = self.parse_unarynum()
        while self.cur().type == "OP" and self.cur().value in ("*","/"):
            op = self.eat("OP").value
            right = self.parse_unarynum()
            node = ("BINOP", op, node, right)
        return node

    def parse_unarynum(self):
        if self.cur().type == "OP" and self.cur().value == "-":
            self.eat("OP")
            node = self.parse_unarynum()
            return ("UNARY", "-", node)
        return self.parse_primary()

    def parse_primary(self):
        tok = self.cur()

        # literals
        if tok.type == "NUMBER":
            self.eat("NUMBER"); return ("NUMBER", tok.value)
        if tok.type == "STRING":
            self.eat("STRING"); return ("STRING", tok.value)
        if tok.type == "ID" and tok.value in ("true","false","nil"):
            val = {"true": True, "false": False, "nil": None}[tok.value]
            self.eat("ID"); return ("CONST", val)

        # array literal [ a, b, c ]
        if tok.type == "OP" and tok.value == "[":
            self.eat("OP","[")
            items = []
            if not (self.cur().type=="OP" and self.cur().value=="]"):
                items.append(self.parse_expr())
                while self.cur().type == "COMMA":
                    self.eat("COMMA")
                    items.append(self.parse_expr())
            self.eat("OP","]")
            return ("ARRAY", items)

        # object literal { k: v, ... }  (expression context)
        if tok.type == "LBRACE":
            self.eat("LBRACE")
            pairs = []
            if self.cur().type != "RBRACE":
                while True:
                    # key can be ID or STRING
                    if self.cur().type == "ID":
                        key = self.eat("ID").value
                    elif self.cur().type == "STRING":
                        key = self.eat("STRING").value
                    else:
                        raise SyntaxError("Expected key in object literal")
                    if self.cur().type == "OP" and self.cur().value == ":":
                        self.eat("OP",":")
                    else:
                        raise SyntaxError("Expected ':' in object literal")
                    val = self.parse_expr()
                    pairs.append((key,val))
                    if self.cur().type == "COMMA":
                        self.eat("COMMA"); continue
                    break
            self.eat("RBRACE")
            return ("DICT", pairs)

        # identifier / chain / calls / indexing
        if tok.type == "ID":
            node = ("VAR", self.eat("ID").value)

            while True:
                # dot access
                if self.cur().type == "OP" and self.cur().value == ".":
                    self.eat("OP",".")
                    attr = self.eat("ID").value
                    node = ("GETATTR", node, attr)
                    continue
                # indexing: [expr]
                if self.cur().type == "OP" and self.cur().value == "[":
                    self.eat("OP","[")
                    idx = self.parse_expr()
                    self.eat("OP","]")
                    node = ("INDEX", node, idx)
                    continue
                # call: (args)
                if self.cur().type == "LPAREN":
                    self.eat("LPAREN")
                    args = []
                    if self.cur().type != "RPAREN":
                        args.append(self.parse_expr())
                        while self.cur().type == "COMMA":
                            self.eat("COMMA"); args.append(self.parse_expr())
                    self.eat("RPAREN")
                    node = ("CALL", node, args)
                    continue
                break
            return node

        if tok.type == "LPAREN":
            self.eat("LPAREN")
            node = self.parse_expr()
            self.eat("RPAREN")
            return node

        raise SyntaxError(f"Unexpected token {tok}")

# ----------------------------- Runtime -----------------------------
class ReturnException(Exception):
    def __init__(self, value): self.value = value

class Environment:
    def __init__(self, parent=None):
        self.parent = parent
        self.vars = {}
    def get(self, name):
        if name in self.vars: return self.vars[name]
        if self.parent: return self.parent.get(name)
        raise NameError(f"Name '{name}' is not defined")
    def set(self, name, value):
        self.vars[name] = value

class Interpreter:
    def __init__(self, script_dir: pathlib.Path | None):
        self.globals = Environment()
        self._ui_root = None
        self._imported = set()  # track imported module real paths
        self.script_dir = script_dir  # folder of running file (for relative imports)

        # builtins
        self.globals.set("print", ("__builtin_print__", None))
        self.globals.set("neko", {"secret": self._snake_game, "magic_8ball": self._magic_8ball})
        self.globals.set("ui", {
            "window": self._ui_window,
            "label":  self._ui_label,
            "button": self._ui_button,
            "input":  self._ui_input,
            "show":   self._ui_show,
            "after":  self._ui_after,
        })
        # file i/o
        self.globals.set("file", {
            "read": lambda path: open(path, "r", encoding="utf-8").read() if pathlib.Path(path).exists() else "",
            "write": lambda path, content: open(path, "w", encoding="utf-8").write(str(content)) or None,
        })
        # http
        self.globals.set("net", {
            "get": self._net_get, "post": self._net_post
        })
        # simple math shorthands
        self.globals.set("pi", math.pi)

    # ---------- UI ----------
    def _ensure_root(self):
        if not self._ui_root:
            raise RuntimeError("ui.window() must be called first!")

    def _ui_window(self, title="Neko Window", width=600, height=400, bg="#222222"):
        self._ui_root = tk.Tk()
        self._ui_root.title(title)
        self._ui_root.geometry(f"{width}x{height}")
        self._ui_root.configure(bg=bg)

    def _ui_label(self, text, color="white", bg=None, clicked=None):
        self._ensure_root()
        lbl = tk.Label(self._ui_root, text=text, fg=color, bg=bg or self._ui_root["bg"])
        lbl.pack(pady=5)
        if clicked:
            lbl.bind("<Button-1>", lambda e: self._call_neko_func(clicked))
        return lbl

    def _ui_button(self, text, color="white", bg="#444444", clicked=None):
        self._ensure_root()
        btn = tk.Button(self._ui_root, text=text, fg=color, bg=bg)
        if clicked:
            btn.config(command=partial(self._call_neko_func, clicked))
        btn.pack(pady=5)
        return btn

    def _ui_input(self, placeholder="", color="black", bg="white"):
        self._ensure_root()
        entry = tk.Entry(self._ui_root, fg=color, bg=bg)
        if placeholder:
            entry.insert(0, placeholder)
        entry.pack(pady=5)
        return entry

    def _ui_after(self, ms, cb):
        self._ensure_root()
        self._ui_root.after(int(ms), lambda: self._call_neko_func(cb))

    def _ui_show(self):
        self._ensure_root()
        self._ui_root.mainloop()

    def _call_neko_func(self, func_ref):
        if isinstance(func_ref, tuple) and func_ref[0] == "FUNC":
            params, body, env = func_ref[1], func_ref[2], func_ref[3]
            local = Environment(env)
            try:
                return self.eval(body, local)
            except ReturnException as r:
                return r.value
        elif callable(func_ref):
            return func_ref()
        else:
            print("[WARN] Invalid callback")

    # ---------- HTTP ----------
    def _net_get(self, url):
        if not requests: raise RuntimeError("requests module not installed")
        r = requests.get(url); return {"status": r.status_code, "text": r.text}
    def _net_post(self, url, data=""):
        if not requests: raise RuntimeError("requests module not installed")
        r = requests.post(url, data=data); return {"status": r.status_code, "text": r.text}

    # ---------- Easter eggs ----------
    def _snake_game(self):
        try:
            subprocess.run([sys.executable, "-c", SNAKE_CODE])
        except Exception as e:
            print("Could not launch Snake game:", e)

    def _magic_8ball(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        print("=== Magic 8-Ball ===\nAsk questions, 'exit' to leave.")
        responses = [
            "Yes, definitely.", "No chance.", "Maybe later.", "Ask again.",
            "The stars say no.", "Absolutely!", "I wouldn’t count on it.",
            "Very likely.", "Try again tomorrow.", "Neko says... meow, yes."
        ]
        while True:
            q = input("\nYour question: ").strip()
            if q.lower() == "exit":
                os.system('cls' if os.name == 'nt' else 'clear')
                print("Exiting Magic 8-Ball...")
                break
            if not q:
                print("Ask something!"); continue
            print("🎱", random.choice(responses))

    # ---------- Eval ----------
    def eval(self, node, env=None):
        if env is None: env = self.globals
        t = node[0]

        if t == "BLOCK":
            res = None
            for stmt in node[1]:
                res = self.eval(stmt, env)
            return res

        if t == "LET":
            _, name, expr = node
            env.set(name, self.eval(expr, env)); return None

        if t == "EXPR_STMT":
            return self.eval(node[1], env)

        if t == "PRINT":
         val = self.eval(node[1], env)
         # Convert dangerous device names to plain strings
         safe = str(val)
         if safe.upper() in ["PRN", "AUX", "NUL", "CON", "COM1", "LPT1"]:
          safe = f"[{safe}]"  # or append a space, like safe + " "
         print(safe)
         return safe


        if t == "NUMBER": return node[1]
        if t == "STRING": return node[1]
        if t == "CONST":  return node[1]
        if t == "VAR":    return env.get(node[1])

        if t == "ARRAY":
            return [self.eval(e, env) for e in node[1]]

        if t == "DICT":
            d = {}
            for k, vexpr in node[1]:
                d[k] = self.eval(vexpr, env)
            return d

        if t == "GETATTR":
            obj = self.eval(node[1], env)
            attr = node[2]
            # string methods
            if isinstance(obj, str):
                return self._string_method(obj, attr)
            # list methods
            if isinstance(obj, list):
                return self._list_method(obj, attr)
            # dict attrs (keys may not be methods)
            if isinstance(obj, dict):
                if attr in obj: return obj[attr]
                # also expose len()
                if attr == "len": return lambda: len(obj)
                raise AttributeError(f"Object has no attribute '{attr}'")
            raise TypeError(f"Cannot access attribute {attr} of {obj}")

        if t == "INDEX":
            arr = self.eval(node[1], env)
            idx = self.eval(node[2], env)
            if isinstance(arr, (list, str, dict)):
                try:
                    return arr[idx]
                except Exception as e:
                    raise RuntimeError(f"Index error: {e}")
            raise TypeError("Indexing only supports list/str/dict")

        if t == "BINOP":
            _, op, ln, rn = node
            a = self.eval(ln, env); b = self.eval(rn, env)
            return self._apply_binop(op, a, b)

        if t == "UNARY":
            _, op, expr = node
            v = self.eval(expr, env)
            if op == "-": return -v
            raise RuntimeError("Unknown unary op")

        if t == "LOR":
            a = self.eval(node[1], env)
            return a or self.eval(node[2], env)
        if t == "LAND":
            a = self.eval(node[1], env)
            return a and self.eval(node[2], env)
        if t == "LNOT":
            return not self.eval(node[1], env)

        if t == "CALL":
            func_node, args_node = node[1], node[2]
            fn = self.eval(func_node, env)
            args = [self.eval(a, env) for a in args_node]
            if callable(fn):
                return fn(*args)
            if isinstance(fn, tuple) and fn[0] == "FUNC":
                _, params, body, fenv = fn
                local = Environment(fenv)
                for p, a in zip(params, args): local.set(p, a)
                try:
                    return self.eval(body, local)
                except ReturnException as r:
                    return r.value
            raise TypeError(f"{fn} is not callable")

        if t == "FNDEF":
            _, name, params, body = node
            env.set(name, ("FUNC", params, body, env)); return None

        if t == "RETURN":
            val = self.eval(node[1], env); raise ReturnException(val)

        if t == "IF":
            if self.eval(node[1], env):
                return self.eval(node[2], env)
            elif node[3]:
                return self.eval(node[3], env)
            return None

        if t == "WHILE":
            while self.eval(node[1], env):
                self.eval(node[2], env)
            return None

        if t == "IMPORT":
            mod = node[1]
            self._import_module(mod, env); return None

        raise RuntimeError(f"Unhandled node type: {t}")

    def _apply_binop(self, op, a, b):
        ops = {"+": a+b, "-": a-b, "*": a*b, "/": a/b,
               "==": a==b, "!=": a!=b, "<": a<b, ">": a>b, "<=": a<=b, ">=": a>=b}
        if op not in ops: raise RuntimeError(f"Unknown op {op}")
        return ops[op]

    def _stringify(self, v):
        if v is None: return "nil"
        return str(v)

    # ---- helpers for methods ----
    def _string_method(self, s, name):
        if name == "len":   return lambda: len(s)
        if name == "upper": return lambda: s.upper()
        if name == "lower": return lambda: s.lower()
        if name == "split": return lambda sep=" ": s.split(sep)
        if name == "strip": return lambda: s.strip()
        return lambda *a: (_ for _ in ()).throw(AttributeError(f"str has no method '{name}'"))

    def _list_method(self, lst, name):
        if name == "len":  return lambda: len(lst)
        if name == "push": return lambda x: lst.append(x)
        if name == "pop":  return lambda: lst.pop()
        return lambda *a: (_ for _ in ()).throw(AttributeError(f"list has no method '{name}'"))

    # ---- imports ----
    def _resolve_module_paths(self, modname: str) -> List[pathlib.Path]:
        candidates = []
        # if modname looks like a path or endswith .neko, try directly
        if any(c in modname for c in ("/","\\",".neko")):
            p = pathlib.Path(modname)
            if not p.suffix: p = p.with_suffix(".neko")
            if not p.is_absolute():
                # relative to script dir first
                if self.script_dir: candidates.append((self.script_dir / p).resolve())
                candidates.append((pathlib.Path.cwd() / p).resolve())
            else:
                candidates.append(p)
        else:
            # search libs by name
            fname = modname + ".neko"
            if self.script_dir: candidates.append((self.script_dir / "libs" / fname).resolve())
            candidates.append((LOCAL_LIBS / fname).resolve())
            candidates.append((HOME_LIBS / fname).resolve())
            candidates.append((CWD_LIBS / fname).resolve())
        return candidates

    def _import_module(self, modname: str, env: Environment):
        # find file
        paths = self._resolve_module_paths(modname)
        for p in paths:
            if p.exists():
                if str(p) in self._imported: return
                code = p.read_text(encoding="utf-8")
                # simple parse+eval in same interpreter (share globals)
                ast = Parser(tokenize(code)).parse()
                self._imported.add(str(p))
                self.eval(ast, env)
                return
        raise FileNotFoundError(f"Module '{modname}' not found in libs or paths.")

# ----------------------------- Runner -----------------------------
def run(code: str, script_dir: pathlib.Path | None):
    tokens = tokenize(code)
    ast = Parser(tokens).parse()
    Interpreter(script_dir).eval(ast)

# ----------------------------- Snake Game -----------------------------
SNAKE_CODE = r'''
import curses, time, random
s = curses.initscr()
curses.curs_set(0)
sh, sw = s.getmaxyx()
w = curses.newwin(sh, sw, 0, 0)
w.keypad(1)
w.timeout(100)
snake = [[sh//2, sw//4]]
food = [sh//2, sw//2]
w.addch(food[0], food[1], curses.ACS_PI)
key = curses.KEY_RIGHT
while True:
    next_key = w.getch()
    key = key if next_key == -1 else next_key
    head = snake[0][:]
    if key == curses.KEY_DOWN: head[0] += 1
    if key == curses.KEY_UP: head[0] -= 1
    if key == curses.KEY_LEFT: head[1] -= 1
    if key == curses.KEY_RIGHT: head[1] += 1
    if head in snake or head[0] in [0, sh] or head[1] in [0, sw]:
        curses.endwin()
        print("Game over! Score:", len(snake)-1); break
    snake.insert(0, head)
    if head == food:
        food = None
        while food is None:
            nf = [random.randint(1, sh-2), random.randint(1, sw-2)]
            food = nf if nf not in snake else None
        w.addch(food[0], food[1], curses.ACS_PI)
    else:
        tail = snake.pop()
        w.addch(tail[0], tail[1], ' ')
    w.addch(snake[0][0], snake[0][1], curses.ACS_CKBOARD)
'''

# ----------------------------- Meow Package Manager -----------------------------
def _read_registry(url: str) -> dict:
    if not requests:
        raise RuntimeError("requests module not installed (needed for meow install)")
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    mapping = {}
    for line in r.text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"): continue
        if "=" in line:
            name, val = line.split("=", 1)
            mapping[name.strip()] = val.strip()
    return mapping

def _download_to(url: str, dest: pathlib.Path):
    if not requests:
        raise RuntimeError("requests module not installed (needed for meow install)")
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    dest.write_text(r.text, encoding="utf-8")

def meow_install(name_or_url: str):
    # decide URL
    if name_or_url.startswith("http://") or name_or_url.startswith("https://"):
        url = name_or_url
        libname = pathlib.Path(url.split("/")[-1]).stem
    else:
        # resolve via registry
        try:
            reg = _read_registry(REGISTRY_URL)
        except Exception as e:
            raise RuntimeError(f"Failed to read registry: {e}")
        if name_or_url not in reg:
            raise RuntimeError(f"Package '{name_or_url}' not found in registry. Provide a URL instead.")
        url = reg[name_or_url]
        libname = name_or_url

    # where to install: BOTH local (interpreter) and global (home)
    for base in (LOCAL_LIBS, HOME_LIBS):
        dest = (base / f"{libname}.neko")
        _download_to(url, dest)
    print(f"Installed '{libname}' to:\n - {LOCAL_LIBS}\n - {HOME_LIBS}")

def meow_uninstall(name: str):
    removed = []
    for base in (LOCAL_LIBS, HOME_LIBS):
        p = base / f"{name}.neko"
        if p.exists():
            p.unlink(); removed.append(str(p))
    if removed:
        print("Removed:\n" + "\n".join(" - "+x for x in removed))
    else:
        print(f"No installs found for '{name}'")

def meow_list():
    libs = set()
    for base in (LOCAL_LIBS, HOME_LIBS):
        if base.exists():
            for f in base.glob("*.neko"):
                libs.add(f.stem + f"  ({'global' if base==HOME_LIBS else 'local'})")
    if not libs:
        print("No libraries installed.")
    else:
        print("Installed libraries:")
        for x in sorted(libs): print(" -", x)

# ----------------------------- Main -----------------------------
def _is_meow_mode(argv):
    # support both: script invoked as "meow ..." OR "python nekolang_interpreter.py install ..."
    if len(argv) >= 2 and argv[1] in ("install","uninstall","list"):
        return True
    # also if the executable name contains 'meow'
    exe = pathlib.Path(argv[0]).name.lower()
    return "meow" in exe

if __name__ == "__main__":
    argv = sys.argv
    # Magic 8-ball quick entry
    if len(argv) > 1 and any(k in " ".join(argv[1:]).lower() for k in ("8ball","fate")):
        Interpreter(script_dir=None)._magic_8ball()
        sys.exit(0)

    # Meow package manager
    if _is_meow_mode(argv):
        if len(argv) < 2:
            print("Usage: meow install <name|url> | meow uninstall <name> | meow list"); sys.exit(1)
        cmd = argv[1]
        try:
            if cmd == "install":
                if len(argv) < 3: raise SystemExit("Usage: meow install <name|url>")
                meow_install(argv[2])
            elif cmd == "uninstall":
                if len(argv) < 3: raise SystemExit("Usage: meow uninstall <name>")
                meow_uninstall(argv[2])
            elif cmd == "list":
                meow_list()
            else:
                print("Unknown meow command. Use: install | uninstall | list")
        except Exception as e:
            print("meow error:", e); sys.exit(1)
        sys.exit(0)

    # Run a .neko program
    if len(argv) < 2:
        print("Usage: python nekolang_interpreter.py <file.neko>")
        sys.exit(1)
    path = pathlib.Path(argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)
    code = path.read_text(encoding="utf-8")
    run(code, script_dir=path.parent)
