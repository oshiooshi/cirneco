import os
import sys
import pegtree as pg
from transcompiler import TransCompiler

class Samoyed(TransCompiler):
    parser: object

    def __init__(self, grammar_file='samoyed.pegtree'):
        TransCompiler.__init__(self)
        peg = pg.grammar(grammar_file)
        self.parser = pg.generate(peg)

    def generate(self, source: str, display=None):
        tree = self.parser(source)
        if display is not None:
            display(tree)
        code = self.visit(tree)
        return repr(code)

try:
    from IPython.display import SVG, display
    from IPython.core.magic import register_cell_magic, register_line_cell_magic
    import pegtree.graph as graph

    def display_tree(tree):
        v = graph.draw_graph(tree)
        v.format = 'SVG'
        display(SVG(v.render()))

    @register_cell_magic
    def samoyed(option: str, source: str):
        cc = Samoyed()
        display = display_tree if '--tree' in option else None
        code = cc.generate(source, display=display)
        if '--code' in option: # ソースコードを表示する
            print(code)
        # 実行する
        try:
            ipy = get_ipython()
            expr = ipy.input_transformer_manager.transform_cell(code)
            expr_ast = ipy.compile.ast_parse(expr)
            expr_ast = ipy.transform_ast(expr_ast)
            executable = ipy.compile(expr_ast, '', 'exec')
            exec(executable)
        except Exception as e:
            print(repr(code))
            print('エラー検出', e.__class__.__name__, repr(e))
            raise

except:
    pass

def main(argv):
    cc = Samoyed()
    if len(argv) == 0:
        import readline
        try:
            while True:
                line = input('>>> ')
                if line == '':
                    print('Bye')
                    sys.exit(0)
                code = cc.generate(line)
                print(code)
        except EOFError:
            pass
    else:
        filename = argv[0]
        with open(filename) as f:
            code = cc.generate(''.join(f.readlines()))
        print(code)

if __name__ == '__main__':
    main(sys.argv[1:])
