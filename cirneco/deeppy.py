import pegtree as pg
import re
from nmt import PyNMT


def deeppy(text):
    ss = []
    for line in text.split('\n'):
        print("line", line)
        # A そこで、B はオプション形式
        matched = re.findall('(.*)(そこで、|そのとき、|ここで、|さらに)(.*)', text)
        print("matched", matched)
        if len(matched) > 0:
            statement = matched[0][0]
            print("statement", statement)
            options = matched[0][2].split('。')
            print("options", options)
            ss.append((statement, filter_options(options)))
            print("\n\n")
            continue

        for statement in line.split('。'):
            if len(statement) > 0:
                options = []
                ss.append((statement, options))
                
        
    return ss
    
    
def filter_options(options):
    ss = []
    for option in options:
        if len(option) == 0:
            continue
        if option.endswith('ことにする'):
            option = option[:-5]
        ss.append('そこで、'+option)
        print("ss", ss)
    return ss

## 前処理

peg = pg.grammar('samoyed.pegtree')
parser = pg.generate(peg, start='NLStatement')

def fix(tree):
    a = [tree.epos_]
    for t in tree:
        a.append(fix(t).epos_)
    for key in tree.keys():
        a.append(fix(tree.get(key)).epos_)
    tree.epos_ = max(a)
    return tree

def preprocess(s):
    tree = parser(s)
    ss = []
    vars = {}
    index = 0
    for t in tree:
        tag = t.getTag()
        if tag == 'NLPChunk' or tag == 'UName':
            ss.append(str(t))
        else:
            key = ('ABCDEFGHIJKLMN')[index]
            key = f'<{key}>'
            vars[key] = '('+str(fix(t))+')'
            ss.append(key)
            index += 1
    return ''.join(ss), vars

def translate(nmt, s):
    ss = []
    for statement, options in deeppy(s):
        s, vars = preprocess(statement)
        s = nmt.translate(s)
        for key in vars:
            s = s.replace(key, vars[key])
        ss.append(s)
    return '\n'.join(ss)


def start_demo(model='model.pt', src_vocab='japanese.pt', tgt_vocab='python.pt'):
    import IPython
    from google.colab import output

    nmt = PyNMT(model, src_vocab, tgt_vocab)
    translate(nmt, 'もしa+1が偶数ならば')

    def convertOLD(text):
        with output.redirect_to_element('#output'):
            try:
                text = translate(nmt, text)
            except Exception as e:
                print(e)
            display(IPython.display.HTML(f'<textarea style="width: 48%; height:100px">{text}</textarea>'))

    def convert(text):
        try:
            text = translate(nmt, text)
            return IPython.display.JSON({'result': text})
        except Exception as e:
            print(e)
        return e

    output.register_callback('notebook.Convert', convert)

    display(IPython.display.HTML('''
    <textarea id="input" style="float: left; width: 48%; height:100px; font-size: large;"></textarea>
    <div id="output">
    <textarea id="output2" style="width: 48%; height:100px; font-size: large;"></textarea>
    </div>
    <script>
      var timer = null;
      document.getElementById('input').addEventListener('input', (e) => {
        var text = e.srcElement.value;
        if(timer !== null) {
          console.log('clear');
          clearTimeout(timer);
        }
        timer = setTimeout(() => {
        //   var parent = document.getElementById('output');
        //   parent.innerHTML='';
          (async function() {
              const result = await google.colab.kernel.invokeFunction('notebook.Convert', [text], {});
              const data = result.data['application/json'];
              const textarea = document.getElementById('output2');
              textarea.textContent = data.result;
           })();
          timer = null;
        }, 400);
      });
    </script>
    '''))