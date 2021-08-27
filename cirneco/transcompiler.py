from pegtree import ParseTree
from pegtree.visitor import ParseTreeVisitor

import astpy 

class TransCompiler(ParseTreeVisitor):
    def __init__(self, ast=astpy):
        ParseTreeVisitor.__init__(self)
        self.ast = ast

    def acceptSource(self, tree: ParseTree):
        return self.ast.PSource(*[self.visit(t) for t in tree])

    # [#Expression e]
    def acceptExpression(self, tree: ParseTree):
        return self.visit(tree[0])

    def acceptNull(self, tree: ParseTree):
        return self.ast.PValue(None)

    def acceptTrue(self, tree: ParseTree):
        return self.ast.PValue(True)

    def acceptFalse(self, tree: ParseTree):
        return self.ast.PValue(False)

    def acceptInt(self, tree: ParseTree):
        value = int(str(tree))
        return self.ast.PValue(value)

    def acceptFloat(self, tree: ParseTree):
        value = float(str(tree))
        return self.ast.PValue(value)

    def acceptDouble(self, tree: ParseTree):
        value = float(str(tree))
        return self.ast.PValue(value)

    def acceptQString(self, tree: ParseTree):
        value = str(tree)[1:-1]
        return self.ast.PValue(value)

    def acceptMultiString(self, tree: ParseTree):
        value = str(tree)[1:-1]
        return self.ast.PValue(value)

    # [#Group [#Int '1']]
    def acceptGroup(self, tree: ParseTree):
        return self.ast.PGroup(self.visit(tree[0]))

    # [#List [#Int '1'][#Int '2']]
    def acceptList(self, tree: ParseTree):
        return self.ast.PList(*[self.visit(x) for x in tree])

    # [#Tuple [#Int '1'][#Int '2']]
    def acceptTuple(self, tree: ParseTree):
        return self.ast.PTuple(*[self.visit(x) for x in tree])

    # [#Name 'a']
    def acceptName(self, tree: ParseTree):
        return self.ast.PVar(str(tree))
    
    #[#VarDecl name: [#Name 'x'] expr: [#QString '"a"']]
    def acceptVarDecl(self, tree: ParseTree):
        left = self.visit(tree.name)
        right = self.visit(tree.expr)
        return self.ast.PBinary(left, '=', right)

    # [#Assignment left: [#IndexExpr recv: [#Name 'a']index: [#Int '1']]right: [#Int '1']]
    def acceptAssignment(self, tree: ParseTree):
        left = self.visit(tree.left)
        right = self.visit(tree.right)
        return self.ast.PBinary(left, '=', right)

    # [#Infix left: [#Name 'a'] name: [#Name '+'] right: [#Int '1']]
    def acceptInfix(self, tree: ParseTree):
        left = self.visit(tree.left)
        op = str(tree.name)
        right = self.visit(tree.right)
        return self.ast.PBinary(left, op, right)

    # [#IfExpr then: [#Name 'a']cond: [#Name 'b']else: [#Name 'c']]
    def acceptIfExpr(self, tree: ParseTree):
        cond = self.visit(tree.cond)
        then = self.visit(tree.then)
        els = self.visit(tree.get('else'))
        return self.ast.PIfExpr(cond, then, els)
    
    # [#IndexExpr recv: [#Name 'a'] index: [#Int '1']]
    def acceptIndexExpr(self, tree: ParseTree):
        recv = self.visit(tree.recv)
        right = self.visit(tree.index)
        return self.ast.PIndex(recv, right)

    # [#SliceExpr recv: [#Name 'a']start: [#Int '1']]
    def acceptSliceExpr(self, tree: ParseTree):
        recv = self.visit(tree.recv)
        start = self.visit(tree.start) if tree.has('start') else self.ast.PEmpty()
        end = self.visit(tree.end) if tree.has('end') else self.ast.PEmpty()
        step = self.visit(tree.step) if tree.has('step') else self.ast.PEmpty()
        return self.ast.PSlice(recv, start, end, step)

    # [#GetExpr recv: [#Name 'a'] name: [#Name 'b']]
    def acceptGetExpr(self, tree: ParseTree):
        recv = self.visit(tree.recv)
        name = str(tree.name)
        return self.ast.PField(recv, name)

    # [#ApplyExpr name: [#Name 'print']params: [#Arguments '']]
    def acceptApplyExpr(self, tree: ParseTree):
        func = self.visit(tree.name)
        params = [self.visit(x) for x in tree.params]
        return self.ast.PApp(func, *params)

    # [#MethodExpr recv: [#Name 'e']name: [#Name 'print']params: [#Arguments '']]
    def acceptMethodExpr(self, tree: ParseTree):
        recv = self.visit(tree.recv)
        name = str(tree.name)
        func = self.ast.PField(recv, name)
        params = [self.visit(x) for x in tree.params]
        return self.ast.PApp(func, *params)

    def acceptIf(self, tree: ParseTree):
        cond = self.visit(tree.cond)
        then = self.visit(tree.then)
        if tree.has('else'):
            return self.ast.PIf(cond, then, tree.get('else'))
        else:
            return self.ast.PIf(cond, then)

    def acceptBlock(self, tree: ParseTree):
        return self.ast.PBlock(*[self.visit(t) for t in tree])

    def acceptPass(self, tree: ParseTree):
        return self.ast.PStatement('pass')

    def acceptNLP(self, tree: ParseTree):
        ss = []
        vars = {}
        index = 0
        for t in tree:
            tag = t.getTag()
            if tag == 'NLPChunk' or tag == 'UName':
                ss.append(str(t))
            elif tag == 'Block':
                block = self.visit(t)
                return self.ast.NLPStatement(''.join(ss), vars, (block,))
            else:
                key = ('ABCDEFGHIJKLMN')[index]
                key = f'<{key}>'
                vars[key] = '('+str(fix(t))+')'
                ss.append(key)
                index += 1
        return self.ast.NLPStatement(''.join(ss), vars)

def fix(tree):
    a = [tree.epos_]
    for t in tree:
        fix(t)
        a.append(t.epos_)
    for key in tree.keys():
        a.append(fix(tree.get(key)).epos_)
    tree.epos_ = max(a)
    return tree