import { Component, h } from 'preact';

import {loadMNISTData,CreatePerceptronNetwork,
  trainBatch,getInputActivations} from './perceptron-mnist-demo';

//
// This file contains all the figures for the post.
// The first few are faked with SVG, but DemoNetworkTrainer
// loads and trains the code in the blog post.
//


// PerceptronFigure
// --
// SVG Diagram of a perceptron
export class PerceptronFigure extends Component<any,any> {
  render() {
    return <svg width={320} height={100} viewBox='0 0 320 100'>
      <circle r={15} cx={145} cy={50} fill='none' stroke='black'/>
      <line x1={50} y1={30} x2={130} y2={50} fill='none' stroke='#aaaa00'/>
      <line x1={50} y1={40} x2={130} y2={50} fill='none' stroke='#aaaa00'/>
      <line x1={50} y1={50} x2={130} y2={50} fill='none' stroke='#aaaa00'/>
      <line x1={50} y1={60} x2={130} y2={50} fill='none' stroke='#aaaa00'/>
      <line x1={50} y1={70} x2={130} y2={50} fill='none' stroke='#aaaa00'/>
      <text x={0} y={55} fill="#0000cc">inputs</text>
      <text x={55} y={90} fill="#aaaa00">weights</text>
      <text x={133} y={53} font-size="12" fill="#aaaa00">sum</text>
      <line x1={160} y1={50} x2={170} y2={50} fill='none' stroke='#99aa00'/>
      <text x={171} y={53} font-size="12" fill="#cc0000">+bias</text>
      <line x1={205} y1={50} x2={215} y2={50} fill='none' stroke='#aa9900'/>
      <circle r={15} cx={230} cy={50} fill='none' stroke='black'/>
      <text x={205} y={90} fill="#0000cc">sigmoid</text>
      <text x={219} y={59} font-size="32">~</text>
      <line x1={245} y1={50} x2={260} y2={50} fill='none' stroke='#00cc00'/>
      <text x={262} y={55} fill="#00cc00">output</text>
    </svg>;
  }
}

// NetworkFigure
// --
// SVG Diagram of a perceptron
function MiniPerceptronFigure(props) {
  return <g {...props}>
    <circle r={15} cx={145} cy={50} fill='none' stroke='black'/>
    <line x1={50} y1={30} x2={130} y2={50} fill='none' stroke='#aaaa00'/>
    <line x1={50} y1={40} x2={130} y2={50} fill='none' stroke='#aaaa00'/>
    <line x1={50} y1={50} x2={130} y2={50} fill='none' stroke='#aaaa00'/>
    <line x1={50} y1={60} x2={130} y2={50} fill='none' stroke='#aaaa00'/>
    <line x1={50} y1={70} x2={130} y2={50} fill='none' stroke='#aaaa00'/>
    <line x1={160} y1={50} x2={170} y2={50} fill='none' stroke='#99aa00'/>
    <circle r={15} cx={185} cy={50} fill='none' stroke='black'/>
    <line x1={200} y1={50} x2={215} y2={50} fill='none' stroke='#00cc00'/>
  </g>
}
function MiniPerceptronLayer(props) {
  return <g {...props}>
    <MiniPerceptronFigure transform="translate(0,0) scale(0.2,0.5)"/>
    <MiniPerceptronFigure transform="translate(0,20) scale(0.2,0.5)"/>
    <MiniPerceptronFigure transform="translate(0,40) scale(0.2,0.5)"/>
    <MiniPerceptronFigure transform="translate(0,60) scale(0.2,0.5)"/>
    <MiniPerceptronFigure transform="translate(0,80) scale(0.2,0.5)"/>
    <MiniPerceptronFigure transform="translate(0,100) scale(0.2,0.5)"/>
  </g>
}
export class NetworkFigure extends Component<any,any> {
  render() {
    return <svg width={300} height={180} viewBox='0 0 300 180'>
      <text x={10} y={95} fill="#0000cc">inputs</text>
      <text x={130} y={15} fill="#aaaa00">layers</text>
      <MiniPerceptronLayer transform="translate(70,20)"/>
      <MiniPerceptronLayer transform="translate(110,20)"/>
      <MiniPerceptronLayer transform="translate(150,20)"/>
      <MiniPerceptronLayer transform="translate(190,20)"/>
      <rect x={116} y={25} width={38} height={140} stroke="#aaaa00" fill="transparent"/>
      <text x={242} y={95} fill="#00cc00">output</text>
    </svg>;
  }
}

// NetworkFigure
// --
// Animated svg simulating an activated perceptron
export class ActivationFigure extends Component<any,any> {
  render() {
    return <svg width={300} height={180} viewBox='0 0 300 180'>
      <text x={10} y={55} fill="#0000cc">inputs</text>
      <circle r={10} cx={85} cy={30} stroke='none' className='afc-i1'/>
      <circle r={10} cx={85} cy={80} stroke='none' className='afc-i2'/>
      <circle r={15} cx={185} cy={50} stroke='none' className='afc-o'/>
      <line x1={95} y1={30} x2={170} y2={50} fill='none' stroke='#00aa00'/>
      <line x1={95} y1={80} x2={170} y2={50} fill='none' stroke='#aa0000'/>
      <text x={110} y={28} fill="#00cc00">+1.0</text>
      <text x={112} y={92} fill="#cc0000">-1.0</text>
      <text x={222} y={55} fill="#00cc00">output</text>
    </svg>;
  }
}


// DemoValueCanvas
// --
// A wrapper around <canvas> tag that automatically redraws
// when the data props are changed. values must be scaled
// to 0.0 to 1.0

export interface DemoValueCanvasProps {
  width: number;
  height: number;
  values: Float32Array;
  color?: any;
  style?: any;
}

export class DemoValueCanvas extends Component<DemoValueCanvasProps,any> {
  // store the canvas element
  el: HTMLCanvasElement;

  // update on new props
  componentWillReceiveProps() {
    if (this.el) {
      setTimeout(this.drawLayerCanvas,1);
    }
  }

  // bind and draw the canvas when the element is inserted
  bindLayerCanvas = (el) => {
    this.el = el;
    if (el) {
      setTimeout(this.drawLayerCanvas,1);
    }
  }

  // draws the canvas
  drawLayerCanvas = () => {
    const {width, height, color, values} = this.props;

    // skip if unbound
    if (!this.el) return;

    // get a 2d canvas context
    const rlCtx = this.el.getContext('2d');
    if (!rlCtx) return;

    // disable antialias
    (rlCtx as any).webkitImageSmoothingEnabled = false;
    // fill with background color
    rlCtx.fillStyle = `rgba(40,40,40,1)`;
    rlCtx.fillRect( 0, 0, width, height );

    // for each layer value, draw the pixel at opacity
    if (values) {
      for (let r = 0; r < height; ++r) {
        for (let c = 0; c < width; ++c) {
          const i = r * width + c;
          const v = values[i] || 0;
          rlCtx.fillStyle = `rgba(${color},${v})`;
          rlCtx.fillRect( c, r, 1, 1 );
        }
      }
    }
  }

  // render!
  render() {
    const {width, height, style} = this.props;
    return (
      <canvas className='demo-value-canvas'
        width={width} height={height}
        style={style}
        ref={this.bindLayerCanvas}
      ></canvas>
    );
  }
}

// DemoLayerRow
// --
// Renders all the weight and activation canvases of a layer
export class DemoLayerRow extends Component<any,any> {
  render() {
    const {layer, layerIdx, inputActivations} = this.props;
    return (
      <div className='demo-layer-row'>
        <div className='demo-layer-label'>layer#{layerIdx}</div>
        <div className='demo-layer-row-canvases'>
          {layer && layer.perceptrons.map((perceptron, perceptronIdx) => {
            const inputSize = layer.perceptrons[perceptronIdx].weights.length;
            // square the layer size
            const laSideLen = Math.max(Math.sqrt(inputSize));
            // get activations
            let weightedActivations = null;
            if (inputActivations) {
              weightedActivations = new Float32Array(inputSize);
              for (let aIdx = 0; aIdx < inputSize; ++aIdx) {
                weightedActivations[aIdx] = Math.max(Math.min(2 * inputActivations[aIdx]
                  * layer.perceptrons[perceptronIdx].weights[aIdx]));
              }
            }
            // adjust weights to within 0.0 to 1.0
            let weightOpacities = new Float32Array(inputSize);
            for (let aIdx = 0; aIdx < inputSize; ++aIdx) {
              weightOpacities[aIdx] = Math.max(Math.min(0.2
                + layer.perceptrons[perceptronIdx].weights[aIdx], 1.0), 0.0);
            }
            return (
              <div className='demo-layer-perceptron-group'>
                <DemoValueCanvas
                  width={laSideLen}
                  height={laSideLen}
                  values={weightOpacities}
                  color='240,255,0'
                />
                <DemoValueCanvas
                  width={laSideLen}
                  height={laSideLen}
                  values={weightedActivations}
                  color='0,255,0'
                />
              </div>
            );
          })}
        </div>
      </div>
    )
  }
}

// DemoNetworkTrainer
// --
// The big kahuna. Loads the data, trains the network in batches,
// and evaluates t10k

const BATCH_SIZE = 10;
const BATCH_DELAY = 10; // milliseconds

export class DemoNetworkTrainer extends Component<any,any> {
  // state for this gui
  state = {
    loading: false,
    loaded: false,
    started: false,
    paused: false,
    trained: false,
    mnistData: null,
    network: CreatePerceptronNetwork(28 * 28, [10]),
    sampleId: null,
    inputActivations: null,
    activations: null,
  }

  // loads the data
  onLoadClick = () => {
    this.setState({loading: true});
    loadMNISTData('t10k').then((mnistData) => {
      this.setState({
        mnistData, loaded: true,
        sampleId: 0,
        inputActivations: getInputActivations(mnistData, 0),
      });
    });
  }

  // creates a new network object to prepare training
  onStartTrainClick = () => {
    const {mnistData} = this.state;
    this.setState({started:true}, this.beginTraining);
  }

  // begins training
  beginTraining = () => {
    // instead of a for loop, we define a function so we can run async
    // and allow the gui to update in-between
    this.processNextBatch(0);
  }

  processNextBatch = (batchSampleId) => {
    const {paused, mnistData, network} = this.state;

    // stop if paused
    if (paused) return;

    // while we have more sampleIds
    if (batchSampleId < mnistData.numberOfSamples) {

      // train this batch
      const {lastSampleId, lastActivations} = trainBatch(network, mnistData, batchSampleId, BATCH_SIZE);

      // update the gui
      this.setState({network,
        sampleId: lastSampleId,
        inputActivations: getInputActivations(mnistData, lastSampleId),
        activations: lastActivations,
      });

      // do the next batch after a short delay
      setTimeout(() => this.processNextBatch(batchSampleId + BATCH_SIZE), BATCH_DELAY);
    } else {
      // all done
      this.setState({trained:true});
    }
  }

  onTogglePauseTraining = () => {
    this.setState({paused: !this.state.paused}, () => {
      if (!this.state.paused) this.beginTraining();
    });
  }

  render() {
    const {loading, loaded, started, trained, paused, mnistData,
      sampleId, activations, inputActivations, network} = this.state;
    return (
      <div className='demo-trainer'>
        {!loaded ? (
            <div className='demo-group' style={{marginTop:20}}>
              <div>Waiting to load data</div>
              <button onClick={this.onLoadClick} disabled={loading}>Load MNIST T10K Data (8MB)</button>
              {loading ?
                <div className='demo-group'>Loading</div>
              : null}
            </div>
        ) :
          <div className='demo-group'>
            <div>Loaded {mnistData.numberOfSamples} Samples</div>
            {started ?
              <div>
                <div>Trained {1 + sampleId} Samples</div>
                <button onClick={this.onTogglePauseTraining}>{paused ? 'Resume' : 'Pause'} Training</button>
              </div>
            :
              <button onClick={this.onStartTrainClick}>Begin Training</button>
            }
          </div>
        }
        <div className='demo-group'>
          <div className='demo-layers'>
            <div className='current-sample'>
              <div className='demo-layer-label'>in</div>
              <DemoValueCanvas
                width={28}
                height={28}
                values={inputActivations}
                color='0,64,255'
              />
            </div>
            {network && network.layers.map((layer,layerIdx) => (
              <DemoLayerRow
                layer={layer}
                layerIdx={layerIdx}
                inputActivations={(layerIdx === 0) ? inputActivations :
                  activations && activations[layerIdx - 1]}
              />
            ))}
            <div className='outputs-sample'>
              <div className='demo-layer-label'>out</div>
              <DemoValueCanvas
                width={1}
                height={10}
                values={activations && activations[activations.length - 1]}
                color='0,255,64'
              />
            </div>
          </div>
        </div>
      </div>
    )
  }
}
