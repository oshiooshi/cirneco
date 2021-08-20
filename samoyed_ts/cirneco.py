import os
import sys
import pegtree as pg
from transcompiler import TransCompiler


class Cirneco(TransCompiler):
    parser: object

    def __init__(self, grammar_file='cirneco.pegtree'):
        TransCompiler.__init__(self)
        peg = pg.grammar(grammar_file)
        self.parser = pg.generate(peg)

    def generate(self, source: str):
        tree = self.parser(source)
        code = self.visit(tree)
        return repr(code)

def main(argv):
    cirneco = Cirneco()
    if len(argv) == 0:
        import readline
        try:
            while True:
                line = input('cir?>>> ')
                if line == '':
                    print('Bye')
                    sys.exit(0)
                code = cirneco.generate(line)
                print(code)
        except EOFError:
            pass
    else:
        filename = argv[0]
        with open(filename) as f:
            code = cirneco.generate(''.join(f.readlines()))
        filename = filename.replace('.py', '.js')
        with open(filename, 'w') as f:
            f.write(code)
            f.write('\n')
        os.system(f'node {filename}')


if __name__ == '__main__':
    main(sys.argv[1:])
