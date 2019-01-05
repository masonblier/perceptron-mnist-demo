import * as preact from 'preact';
const { Component, h } = preact;
(window as any).preact = preact;
(window as any).h = preact.h;

import * as Figures from './figures';


// This file loads the source code and renders the blog post,
// it's not as well documented.

export function parseCommentToTsx(text) {
  if (text.indexOf('//=') === 0) {
    const figureType = text.substring(3);
    const Figure = Figures[figureType];
    if (!Figure) throw new Error("figure not found "+figureType);
    return <Figure/>;
  } else if (text.indexOf('//') === 0) {
    return <div>{text.substr(2).trim().split(' ').map((word) => (
        (/^\[.+\]\(.+\)$/g.test(word)) ? (() => {
          const firstCloseBracket = word.indexOf(']');
          const text = word.substring(1,firstCloseBracket);
          const url = word.substring(firstCloseBracket+2,word.length-1);
          return (
            <span><a href={url}>{text}</a> </span>
          );
        })() :
          <span>{word} </span>
      ))}</div>;
  } else {
    // empty?
    return <div></div>;
  }
}
const TS_KEYWORDS_RGX = /(for)|(while)|(if)|(else)|(new)|(return)|(var)|(let)|(const)|(function)|(class)|(extends)|(export)|(import)|(interface)/g;
export function parseCodeToTsx(text) {
  // if line is comment line
  if (/ *\/\//g.test(text)) {
    return <div style={{color:'gray'}}>{text}</div>
  }

  // we replace keywords, fns, and variable names
  // we special strings for easier replacement later
  let annotatedText = text.replace(TS_KEYWORDS_RGX, (keywordText) => {
    return ':!~!:'+'$color:#00cc00~!:'+keywordText+':!~!:';
  });
  // fn calls or defs
  annotatedText = annotatedText.replace(/([a-zA-Z_][a-zA-Z0-9_]+)\(/g, (fm, fnName) => {
    return ':!~!:'+'$color:#00ccff~!:'+fnName+':!~!:'+'(';
  });
  return (
    <div>
      {annotatedText.split(':!~!:').map((word) => (
        word.indexOf("$color:") === 0 ?
          <span style={{color:word.substring(7,14)}}>{word.substr(17)}</span>
        :
          word.length > 0 ?
            <span>{word}</span>
          : null
      ))}
    </div>
  );
}

// Layout root preact element
export class Layout extends Component<any,any> {
  state = {
    parsedGroups: [],
  }

  componentDidMount() {
    this.loadSource();
  }

  loadSource = () => {
    // fetch the source code text
    fetch('./perceptron-mnist-demo.tsx').then((r) => r.text()).then((sourceText) => {

      // array to hold the groups
      const parsedGroups = [];

      // a group is a block of comments followed by a block of code
      function pushNewGroup() {
        // only push a new group if existing group is not empty
        const tg = topGroup();
        if ((!tg) || (tg.comments.length > 0) || (tg.code.length > 0)) {
          // if previous tg had code, remove excess blank lines
          if (tg && tg.code.length > 0) {
            while ((tg && tg.code.length > 0)
              && ((tg.code[tg.code.length - 1].children.length === 1)
                && (tg.code[tg.code.length - 1].children[0].length === 0))
            ) {
              tg.code.pop();
            }
          }
          parsedGroups.push({comments:[],code:[]});
        }
      }
      function topGroup() {
        return parsedGroups[parsedGroups.length - 1];
      }

      // for each line, it can be comment, code, or empty,
      // so track the lastType to detect when this changes.
      // empty does not change the lastType state.
      let lastType = null;
      // start with one group
      pushNewGroup();
      sourceText.split(/\r?\n/g).forEach((line) => {
        // comment. dont care about indented comments
        if (line.indexOf('//') === 0) {
          // if last line was code, start a new group
          if (lastType === 'code') {
            pushNewGroup();
          }

          let skip = false;

          // if this line is a header-marker
          if ((line.indexOf('// ==') === 0)) {
            if (topGroup().comments.length > 0) {
              const cmts = topGroup().comments;
              // wrap previous line in h1
              cmts[cmts.length - 1] = <h1>{cmts[cmts.length - 1]}</h1>;
            }
            skip = true;

          // sub header market
          } else if ((line.indexOf('// --') === 0)) {
            if (topGroup().comments.length > 0) {
              const cmts = topGroup().comments;
              cmts[cmts.length - 1] = <h2>{cmts[cmts.length - 1]}</h2>;
            }
            skip = true;
          }

          if (!skip) {
            // add comment to group
            topGroup().comments.push(parseCommentToTsx(line));
          }

          // update lastType
          lastType = 'comment';

        // else, if not empty, then it's code
        } else if (line.trim().length > 0) {
          // add code to group
          topGroup().code.push(parseCodeToTsx(line));

          // update lastType
          lastType = 'code';

        // else empty
        } else {
          // empty between code is preserved
          if (lastType === 'code') {
            topGroup().code.push(parseCodeToTsx(line));
          }
        }
      });

      this.setState({parsedGroups})
    });
  }

  render() {
    const {parsedGroups} = this.state;
    return (
      <div className='blog-layout'>
        {parsedGroups.map((group) => (
          <div className='layout-group'>
            {group.comments.length > 1 ?
              <div className='layout-comments'>
                {group.comments}
              </div>
            : null}
            {group.code.length > 1 ?
              <div className='layout-code'>
                {group.code}
              </div>
            : null}
          </div>
        ))}
      </div>
    );
  }
}

// This is called from the HTML file to start the preact app
export function render(containerEl) {
  if (containerEl != null) {
    preact.render(<Layout />, containerEl, containerEl.lastElementChild || undefined);
  }
}
