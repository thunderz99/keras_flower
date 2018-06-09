import React from 'react';
import logo from './logo.svg';
import './App.css';
import Dropzone from 'react-dropzone'
import fetch from 'node-fetch'

class App extends React.Component {
  constructor() {
    super()
    this.state = { files: [] }
  }

  onDrop(files) {

    var formData = new FormData();
    for (const file of files) {
        formData.append('file[]', file);    
    }

    fetch('/flask/uploader', {
        method:'POST',
        body:formData   
    }).then( res => {
        res.json().then( json => {
          console.log('res:' + json)
          this.setState({files: json})
        })
    }).catch( e => {
        console.log('Error',e);
    });


    this.setState({
      files
    });
  }

  render() {

    const dropStyle = {width: '100%'}
    const imgWidth = '300px'

    let links = [
      {
        name: 'あじさい',
        url: 'https://www.google.co.jp/search?q=%E3%81%82%E3%81%98%E3%81%95%E3%81%84+%E3%82%A4%E3%83%A9%E3%82%B9%E3%83%88+%E7%99%BD%E9%BB%92',
      },
      {
        name: 'すみれ',
        url: 'https://www.google.co.jp/search?q=%E3%81%99%E3%81%BF%E3%82%8C+%E3%82%A4%E3%83%A9%E3%82%B9%E3%83%88+%E7%99%BD%E9%BB%92'
      },
      {
        name: 'さくら',
        url: 'https://www.google.co.jp/search?q=%E3%81%95%E3%81%8F%E3%82%89+%E3%82%A4%E3%83%A9%E3%82%B9%E3%83%88+%E7%99%BD%E9%BB%92'
      },
      {
        name: 'すいせん',
        url: 'https://www.google.co.jp/search?q=%E3%81%99%E3%81%84%E3%81%9B%E3%82%93+%E3%82%A4%E3%83%A9%E3%82%B9%E3%83%88+%E7%99%BD%E9%BB%92'
      }

    ]

    return (
      <section>
        <div className="App">
          <header className="App-header">
            <img src={logo} className="App-logo" alt="logo" />
            <h1 className="App-title">花イラスト認識Demo</h1>
          </header>
        </div>
        <div style={{height: '10px'}} />
        <div className="dropzone" style={dropStyle}>
          <Dropzone onDrop={this.onDrop.bind(this)}
            accept="image/*"
            style={{
              "width": "90%",
              "height": 150,
              "borderWidth": 2,
              "borderColor": "#666",
              "borderStyle": "dashed",
              "borderRadius": 5,
              "margin": "auto"
            }}
          >
            <p>花イラストの画像を選択してください</p>
          </Dropzone>
        </div>
        <div>
          <h4>テスト用画像ダウンロードはこちら</h4>
          <ul>
            {
            links.map(l => <li><a target="_blank" href={l.url}>{l.name}</a></li>)
            }
          </ul>
        </div>
        <aside>
          <h3>結果</h3>
          <ul>
            {
              this.state.files.map(f => <li key={f.filename}><img width={imgWidth} src={f.url} /><h3>{f.result}</h3></li>)
            }
          </ul>
        </aside>
      </section>
    );
  }
}

<App />

export default App;
