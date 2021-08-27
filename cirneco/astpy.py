
EMPTY = () # 空のタプル

class Env(dict):
    indent: int
    def __init__(self):
        self.indent = 0
    

class PExpr(object):  # 式
    name: str
    params: tuple

    def __init__(self, name='', params=EMPTY):
        self.name = name
        self.params = params

    def emit(self, env, buffers):
        pass

    def __repr__(self):
        buffers = []
        self.emit(Env(), buffers)
        return ''.join(buffers)

    # def __lt__(self, a):
    #     return id(self) < id(a)

    def __len__(self):
        return len(self.params)

    def __getitem__(self, index):
        return self.params[index]

ESC = str.maketrans({'\n': '\\n', '\t': '\\t', '"': '\\"'})

class PValue(PExpr): # 値
    value: object

    def __init__(self, value):
        PExpr.__init__(self)
        self.value = value

    def emit(self, env, buffers):
        value = self.value
        if isinstance(value, str):
            buffers.append('"')
            buffers.append(value.translate(ESC))
            buffers.append('"')
        else:
            buffers.append(str(value))

class PVar(PExpr):  # 変数

    def __init__(self, name):
        PExpr.__init__(self, name)

    def emit(self, env, buffers):
        buffers.append(self.name)

def e(x):
    if isinstance(x, PExpr):
        return x
    if isinstance(x, str):
        return PVar(x)
    return PValue(x)


class PBinary(PExpr):
    MAP = {
        'and': 'and',
        'or': 'or',
    }
    def __init__(self, left, op, right):
        PExpr.__init__(self, op.strip(), (e(left), e(right)))

    def emit(self, env, buffers):
        self[0].emit(env, buffers)
        op = PBinary.MAP.get(self.name, self.name)
        buffers.append(f' {op} ')
        self[1].emit(env, buffers)

class PUnary(PExpr):
    MAP = {
        'not': 'not ',
    }

    def __init__(self, op, expr):
        PExpr.__init__(self, op, (e(expr),))

    def emit(self, env, buffers):
        buffers.append(PUnary.MAP.get(self.name, self.name))
        self.params[0].emit(env, buffers)

class PIfExpr(PExpr):
    def __init__(self, *es):
        PExpr.__init__(self, '{}', tuple(e(x) for x in es))

    def emit(self, env, buffers):
        self.params[1].emit(env, buffers)
        buffers.append(' if ')
        self.params[0].emit(env, buffers)
        buffers.append(' else ')
        self.params[2].emit(env, buffers)


class PField(PExpr):
    def __init__(self, expr, name: str):
        PExpr.__init__(self, name, (e(expr), ))

    def emit(self, env, buffers):
        self.params[0].emit(env, buffers)
        buffers.append('.')
        buffers.append(self.name)

class PIndex(PExpr):
    def __init__(self, expr, index):
        PExpr.__init__(self, [], (e(expr), e(index)))

    def emit(self, env, buffers):
        self.params[0].emit(env, buffers)
        buffers.append('[')
        self.params[1].emit(env, buffers)
        buffers.append(']')

class PEmpty(PExpr):
    def __init__(self):
        PExpr.__init__(self)

    def emit(self, env, buffers):
        pass

class PSlice(PExpr):
    def __init__(self, expr, index):
        PExpr.__init__(self, '[]', (e(expr), e(index)))

    def emit(self, env, buffers):
        self.params[0].emit(env, buffers)
        buffers.append('[')
        self.params[1].emit(env, buffers)
        buffers.append(':')
        self.params[2].emit(env, buffers)
        if len(self.params[3]) > 3:
            self.params[3].emit(env, buffers)
        buffers.append(']')


class PApp(PExpr):

    def __init__(self, *es):
        PExpr.__init__(self, '', tuple(e(x) for x in es))

    def emit(self, env, buffers):
        self.params[0].emit(env, buffers)
        buffers.append('(')
        for i, p in enumerate(self.params[1:]):
            if i > 0:
                buffers.append(',')
            p.emit(env, buffers)
        buffers.append(')')

class PGroup(PExpr):

    def __init__(self, expr):
        PExpr.__init__(self, "(,)", tuple(e(expr)))

    def emit(self, env, buffers):
        buffers.append('(')
        self.params[0].emit(env, buffers)
        buffers.append(')')

class PSeq(PExpr):

    def __init__(self, *es):
        PExpr.__init__(self, ",", tuple(e(x) for x in es))

    def emit(self, env, buffers):
        for i, p in enumerate(self.params):
            if i > 0:
                buffers.append(',')
            p.emit(env, buffers)

class PTuple(PExpr):

    def __init__(self, *es):
        PExpr.__init__(self, "(,)", tuple(e(x) for x in es))

    def emit(self, env, buffers):
        buffers.append('(')
        for i, p in enumerate(self.params):
            if i > 0:
                buffers.append(',')
            p.emit(env, buffers)
        if len(self.params) == 1:
            buffers.append(',')
        buffers.append(')')

class PList(PExpr):

    def __init__(self, *es):
        PExpr.__init__(self, "[,]", tuple(e(x) for x in es))

    def emit(self, env, buffers):
        buffers.append('[')
        for i, p in enumerate(self.params):
            if i > 0:
                buffers.append(',')
            p.emit(env, buffers)
        buffers.append(']')

class PKeyValue(PExpr):

    def __init__(self, key, value):
        PExpr.__init__(self, (e(key), e(value)))

    def emit(self, env, buffers):
        self.params[0].emit(env, buffers)
        buffers.append(':')
        self.params[1].emit(env, buffers)


class PDict(PExpr):

    def __init__(self, *es):
        PExpr.__init__(self, "{,}", tuple(e(x) for x in es))

    def emit(self, env, buffers):
        buffers.append('{')
        for i, p in enumerate(self.params):
            if i > 0:
                buffers.append(',')
            p.emit(env, buffers)
        buffers.append('}')

class PStatement(PExpr):

    def __init__(self, pred, *es):
        PExpr.__init__(self, pred, tuple(e(x) for x in es))

    def emit(self, env, buffers):
        buffers.append(self.name)
        if len(self.params) > 0:
            buffers.append(' ')
            self.params[0].emit(env, buffers)

class PSource(PExpr):
    def __init__(self, *es):
        PExpr.__init__(self, '', tuple(e(x) for x in es))

    def emit(self, env, buffers):
        for i, p in enumerate(self.params):
            if i > 0:
                buffers.append('\n')
            p.emit(env, buffers)

class PBlock(PExpr):
    def __init__(self, *es):
        PExpr.__init__(self, '{}', tuple(e(x) for x in es))

    def emit(self, env, buffers):
        indent = env.get('@indent', '\n')
        env['@indent'] = indent+'\t'
        for i, p in enumerate(self.params):
            buffers.append(env['@indent'])
            p.emit(env, buffers)
        env['@indent'] = indent

class PIf(PExpr):
    def __init__(self, *es):
        PExpr.__init__(self, '{}', tuple(e(x) for x in es))

    def emit(self, env, buffers):
        buffers.append('if ')
        self.params[0].emit(env, buffers)
        buffers.append(':')
        self.params[1].emit(env, buffers)
        if len(self.params) > 2:
            buffers.append(env.get('@indent', '\n'))
            buffers.append('else:')
            self.params[2].emit(env, buffers)

class PWhile(PExpr):
    def __init__(self, *es):
        PExpr.__init__(self, '{}', tuple(e(x) for x in es))

    def emit(self, env, buffers):
        buffers.append('while ')
        self.params[0].emit(env, buffers)
        buffers.append(':')
        self.params[1].emit(env, buffers)

class PFor(PExpr):
    def __init__(self, *es):
        PExpr.__init__(self, '{}', tuple(e(x) for x in es))

    def emit(self, env, buffers):
        buffers.append('for ')
        self.params[0].emit(env, buffers)
        buffers.append(' in ')
        self.params[1].emit(env, buffers)
        buffers.append(':')
        self.params[2].emit(env, buffers)

class PForRange(PExpr):

    def __init__(self, *es):
        PExpr.__init__(self, '{}', tuple(e(x) for x in es))

    def emit(self, env, buffers):
        buffers.append('for ')
        self.params[0].emit(env, buffers)
        buffers.append(' in range(')
        self.params[1].emit(env, buffers)
        if len(self.params) > 2:
            buffers.append(',')
            self.params[3].emit(env, buffers)
        if len(self.params) > 3:
            buffers.append(',')
            self.params[4].emit(env, buffers)
        buffers.append('): ')
        self.params[-1].emit(env, buffers)

try:
    import nmt
    model = nmt.NMT()
    model.load('./all-model.pt')
    print('Programming AI is supported')

    class NLPStatement(PExpr):

        def __init__(self, statement, vars, block=EMPTY):
            PExpr.__init__(self, statement, block)
            self.vars = vars

        def emit(self, env, buffers):
            s = model.translate(self.name).strip()
            for key in self.vars.keys():
                s = s.replace(key, self.vars[key])
            buffers.append(s + "#"+self.name)
            if len(self.params) == 1: #block
                self.params[0].emit(env, buffers)

except Exception as e:
    print('Programming AI unsupported', e)

if __name__ == '__main__':
    print(PIf(PBinary('a', '=', 1), 'pass', 'pass'))
    print(PApp('print'))
    print(PApp('print', 1))