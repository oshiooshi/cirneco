# トランスコンパイラを作る？トランスパイラを作る
# mainが呼ばれる、mainのargが0だとなんとか

import os
import sys
import pegtree as pg
from pegtree import ParseTree
from pegtree.visitor import ParseTreeVisitor


class Transpiler(ParseTreeVisitor):
    buffer: list
    parser: object

    def __init__(self, grammar_file='cirneco.pegtree'):
        ParseTreeVisitor.__init__(self)
        self.buffer = []
        peg = pg.grammar(grammar_file)
        self.parser = pg.generate(peg)

    def push(self, s: str):
        self.buffer.append(s)

    def generate(self, source: str):
        tree = self.parser(source)
        self.buffer = []
        self.visit(tree)
        return ''.join(self.buffer)
        
    def acceptSource(self, tree: ParseTree):
        # self.visit(tree[0]) # 0番目のtreeを処理（最初の子ノードの処理）
        for child in tree:
            self.visit(child)
            self.push('\n')     # 改行を入れておかないと.jsで改行されない

    def acceptExpression(self, tree: ParseTree):
        # self.visit(tree[0]) # 0番目のtreeを処理（最初の子ノードの処理）
        for child in tree:
            self.visit(child)        

    def acceptApplyExpr(self, tree: ParseTree):
        self.visit(tree.name) # nameというラベル付がされている
        self.visit(tree.params) # paramsというラベル付がされている
    
    def acceptArguments(self, tree: ParseTree):
        c = 0
        self.push('(')
        for child in tree:
            if c != 0:
                self.push(',')
            self.visit(child)
            c += 1
        self.push(')')
    
    def acceptVarDecl(self, tree: ParseTree):
        self.visit(tree.name)
        self.push('=')
        self.visit(tree.expr)
    
    def acceptInfix(self, tree: ParseTree):
        self.visit(tree.left)
        self.visit(tree.name)
        self.visit(tree.right)

    def acceptName(self,tree: ParseTree):
        if str(tree) == 'print':
            self.push('console.log') # bufferに入る
        else:
            self.push(str(tree))

    def acceptQString(self,tree: ParseTree):
        self.push(str(tree)) # bufferに入る
        # 末端だからvisitする必要はない、bufferにpushしてる  

    def acceptInt(self,tree: ParseTree):
        self.push(str(tree))   


def main(argv):
    cirneco = Transpiler()
    if len(argv) == 0:
        import readline
        try:
            while True:
                line = input('cir?>>> ')    #ファイルがなければ
                if line == '':
                    print('Bye')
                    sys.exit(0)
                code = cirneco.generate(line)
                print(code)
        except EOFError:
            pass
    else:
        filename = argv[0]      # python3 cirneco.py hello.pyで実行可能
        with open(filename) as f:
            code = cirneco.generate(''.join(f.readlines()))
        filename = filename.replace('.py', '.js')
        with open(filename, 'w') as f:
            f.write(code)
            f.write('\n')
        os.system(f'node {filename}')


if __name__ == '__main__':
    main(sys.argv[1:])
