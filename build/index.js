// perceptron-mnist-demo
// ==
define("perceptron-mnist-demo", ["require", "exports"], function (require, exports) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    // AKA Solving(ish) MNIST from scratch in your browser!
    // By [masonblier](https://github.com/masonblier) 2019-01-05
    // To understand the mathematics of neural networks more intuitively, I wanted
    // to explore a very simple network that could give good visualizations but
    // still run in a browser in real-time. Perceptrons are one of the simplest
    // and oldest neural-net algorithms, while their performance is limited, they
    // are simple enough to implement from scratch in a blog post.
    // For a real project you probably should use a well-tested library, such as
    // [brainjs](https://github.com/BrainJS/brain.js) which some of this code is based on
    // maths
    // ==
    // Lets get some of the basic math functions defined first.
    // sigmoid
    // --
    // The sigmoid function is a way of restricting values between -1.0 and 1.0.
    // You could experiment with other functions instead, such as relu.
    function sigmoid(n) {
        return 1 / (1 + Math.exp(-n));
    }
    // mse
    // --
    // mean squared error for returning a positive error-value between 0.0 and 1.0.
    // It takes a "Float32Array", which is JavaScript-language for float[]
    function mse(errors) {
        let sum = 0;
        for (let errorIdx = 0; errorIdx < errors.length; ++errorIdx) {
            sum += Math.pow(errors[errorIdx], 2);
        }
        return sum / errors.length;
    }
    // randomWeight
    // --
    // Our weights need to be limited, we can do that by linear-scaling the
    // `Math.random()` value, which is 0.0 to 1.0 by default. You will see
    // this pattern often in this file.
    const MAX_WEIGHT = 0.4;
    const MIN_WEIGHT = -0.4;
    function randomWeight() {
        return Math.random() * (MAX_WEIGHT - MIN_WEIGHT) + MIN_WEIGHT;
    }
    // CreatePerception
    // --
    // Creates a single perception. We need the size of the lower layer to know
    // how many weights to store.
    function CreatePerceptron(sizeOfLowerLayer) {
        // One weight for each item of the lower layer. We need to intiate our
        // perceptron randomly to help it converge.
        const weights = new Float32Array(sizeOfLowerLayer);
        for (let weightIdx = 0; weightIdx < sizeOfLowerLayer; ++weightIdx) {
            weights[weightIdx] = randomWeight();
        }
        // A random bias
        const bias = randomWeight();
        // Our Perceptron!
        return { bias, weights };
    }
    // To create the network, we need to prepopulate the layers. We also
    // need to know the size of the input layer, to know how many weights
    // each perceptron of the first hidden layer (or output layer if only one)
    // needs to have.
    //=NetworkFigure
    function CreatePerceptronNetwork(sizeOfInput, otherLayerSizes) {
        // we need at least one layer for output
        if (otherLayerSizes.length < 1) {
            throw new Error("otherLayerSizes requires at least one layer");
        }
        // we build each layer from the sizes list
        const layers = otherLayerSizes.map((layerSize, layerIdx) => {
            // each perceptron needs one weight for each item in the lower layer.
            // but if we are the lowest layer, then we use the input size
            const sizeOfLowerLayer = (layerIdx === 0) ? sizeOfInput : otherLayerSizes[layerIdx - 1];
            // build and init our array of perceptrons
            const perceptrons = new Array(layerSize);
            for (let perceptronIdx = 0; perceptronIdx < layerSize; ++perceptronIdx) {
                perceptrons[perceptronIdx] = CreatePerceptron(sizeOfLowerLayer);
            }
            // Our PerceptronLayer!
            return { perceptrons };
        });
        // Our PerceptronNetwork!
        return { layers };
    }
    exports.CreatePerceptronNetwork = CreatePerceptronNetwork;
    // Activate!
    // ==
    // Now that we have the structure of the network, we can run inputs through it
    // and get outputs. Of course, we will need to train it before it can get things
    // correct, but it's necessary to use these functions in our training functions,
    // and these are easier to understand, so let's write them first.
    //=ActivationFigure
    // getLayerActivations
    // --
    // At the core of it all is propagating the activation values through each layer.
    // A layer gets its activations from each dendrite, reaching down, and pulling
    // the value up, scaling it by the weight, and feeding our sum. Then our sum
    // is shifted by the bias, pushing us up, or down into the dark, never to
    // activate again.
    // Takes the layer and activations to feed as inputs, and returns the outputs
    // activations of the layer.
    function getLayerActivations(layer, inputActivations) {
        // Our output is a float[] the size of our layer
        const output = new Float32Array(layer.perceptrons.length);
        // for each perceptron in our layer
        for (let perceptronIdx = 0; perceptronIdx < layer.perceptrons.length; ++perceptronIdx) {
            // we could start the weight at the bias, save some math,
            // but then my clever joke wouldn't have worked. (it didn't anyways)
            let sum = 0.0;
            // for each weight in the perceptron
            for (let weightIdx = 0; weightIdx < layer.perceptrons[perceptronIdx].weights.length; ++weightIdx) {
                sum += layer.perceptrons[perceptronIdx].weights[weightIdx] * inputActivations[weightIdx];
            }
            // shift it
            sum += layer.perceptrons[perceptronIdx].bias;
            // We sigmoid our sum instead of averaging.
            // I probably should just ask someone about this instead of writing
            // useless comments
            output[perceptronIdx] = sigmoid(sum);
        }
        return output;
    }
    function getNetworkActivations(network, inputActivations) {
        // container array to hold each layer of activations
        const networkActivations = new Array(network.layers.length);
        // for each layer in the network
        for (let layerIdx = 0; layerIdx < network.layers.length; ++layerIdx) {
            // the previous activations are either from the lower layer, or the input
            const lowerActivations = (layerIdx === 0) ? inputActivations : networkActivations[layerIdx - 1];
            // and simply calculate. the result of this will become the lowerActivations
            // of the next iteration
            networkActivations[layerIdx] = getLayerActivations(network.layers[layerIdx], lowerActivations);
        }
        // the last LayerActivations in this array is the output of the network!
        return networkActivations;
    }
    exports.getNetworkActivations = getNetworkActivations;
    // Training
    // ==
    // Now we are ready to train the network. The core of this process is the
    // trainPattern function, which adjusts the network weights to push the output
    // just a bit closer to the expectedOutputs. Eventually we will do this many times,
    // repeatedly and randomly, to train the network.
    // trainPattern
    // --
    // Trains the network for a single input/output mapping. This is a three-step
    // process, by first getting the current activations as-is, then calculating
    // all the errors, and deltas we wish to make, and finally adjusting the weights
    // relative to those deltas. More details in each function below.
    // We return the mean squared error of the output activations.
    function trainPattern(network, inputActivations, expectedOutputs) {
        // activate all current layers with the input
        const networkActivations = getNetworkActivations(network, inputActivations);
        // our actualOutputs is the top layer of activations
        const actualOutputs = networkActivations[networkActivations.length - 1];
        // calculate delta and layer error values
        const { deltas, layerErrors } = calculateDeltasAndErrors(network, actualOutputs, expectedOutputs);
        // adjust network weights!
        // we do this in-place because creating new float[]s is expensive
        adjustWeights(network, deltas, inputActivations, networkActivations);
        // return the mean squared error of the output layer
        return { networkError: mse(layerErrors[layerErrors.length - 1]), networkActivations };
    }
    function calculateDeltasAndErrors(network, actualOutputs, expectedOutputs) {
        // array for each layer of deltas
        const deltas = new Array(network.layers.length);
        // array for each layer of errors
        const layerErrors = new Array(network.layers.length);
        // iterate layers output-to-input
        for (let layerIdx = network.layers.length - 1; layerIdx >= 0; --layerIdx) {
            // shorthand for the perceptrons of this layer
            const layerPerceptrons = network.layers[layerIdx].perceptrons;
            // array for the deltas of this layer
            deltas[layerIdx] = new Float32Array(layerPerceptrons.length);
            // array for the errors of this layer
            layerErrors[layerIdx] = new Float32Array(layerPerceptrons.length);
            // for each perceptron in this layer
            for (let perceptronIdx = 0; perceptronIdx < layerPerceptrons.length; ++perceptronIdx) {
                // start the error at zero
                let errorSum = 0.0;
                // if this is the top layer, compare the outputs
                if (layerIdx === (network.layers.length - 1)) {
                    errorSum = expectedOutputs[perceptronIdx] - actualOutputs[perceptronIdx];
                }
                else {
                    // Otherwise, calculate the error by deriving it. We do this by tracing
                    // each weight to the upper layer perceptrons, and using the chain rule on
                    // the activation equation in getLayerActivations. Everything is linear,
                    // so the derivative is just a scalar multiplication.
                    // I don't understand it perfectly yet, but 3blue1brown has a good video.
                    const deltasFromUpperLayer = deltas[layerIdx + 1];
                    for (let upperPerceptronIdx = 0; upperPerceptronIdx < deltasFromUpperLayer.length; ++upperPerceptronIdx) {
                        // get the upper layer perceptron that this delta corresponds to
                        const upperPerceptron = network.layers[layerIdx + 1].perceptrons[upperPerceptronIdx];
                        // use the weight that connects the upper perceptron to this one
                        const weightFromDendrite = upperPerceptron.weights[perceptronIdx];
                        // and sum the scaled deltas to calculate the error of our layer
                        errorSum += deltasFromUpperLayer[upperPerceptronIdx] * weightFromDendrite;
                    }
                }
                // save the error of this layer
                layerErrors[layerIdx][perceptronIdx] = errorSum;
                // the delta is used below when adjusting the weights
                // TODO why output_p * (1 - output_p) ?
                deltas[layerIdx][perceptronIdx] = errorSum * actualOutputs[perceptronIdx]
                    * (1 - actualOutputs[perceptronIdx]);
            }
        }
        return { deltas, layerErrors };
    }
    // adjustWeights
    // --
    // And finally, we can update the weights
    //
    // learning rate controls how strongly the weights are influenced each training.
    // this is one of the hyperparameters of our network, feel free to play with it.
    const LEARNING_RATE = 0.3;
    function adjustWeights(network, deltas, inputActivations, networkActivations) {
        // for each layer
        for (let layerIdx = 0; layerIdx < network.layers.length; ++layerIdx) {
            // get the array of perceptrons for this layer
            const layerPerceptrons = network.layers[layerIdx].perceptrons;
            // get the input activations for this layer
            const lowerActivations = (layerIdx === 0) ? inputActivations : networkActivations[layerIdx - 1];
            // for each perceptron in this layer
            for (let perceptronIdx = 0; perceptronIdx < layerPerceptrons.length; ++perceptronIdx) {
                // get the delta for this perceptron
                const perceptronDelta = deltas[layerIdx][perceptronIdx];
                // for each of the weights of our perceptron
                for (let weightIdx = 0; weightIdx < lowerActivations.length; ++weightIdx) {
                    // adjust the weight according to delta, input activation, and learning rate
                    const change = perceptronDelta * lowerActivations[weightIdx] * LEARNING_RATE;
                    // adjust!
                    layerPerceptrons[perceptronIdx].weights[weightIdx] += change;
                }
            }
        }
    }
    // Running the program
    // ==
    // Nice! All the Perceptron stuff is done. Only 356 lines (mostly comments).
    // Not sure if it helped your understanding, but it helped mine to write it.
    // To wrap up the process, we create a function that will train the network
    // from all the items in an MNIST data set. Loading the data is further below,
    // but using getInputActivations and getOutputActivations, we get the data
    // we need for each training sample.
    // trainBatch
    // --
    // Trains the batch in a loop, either approaching a minimal error, or
    // exhausting the iteration limit.
    const MAX_ITERATIONS = 200; // usually use more, but it gets slow
    const ERROR_THRESHOLD = 0.005;
    function trainBatch(network, mnistData, batchStartId, batchSize) {
        // store lastSampleId and lastActivations for GUI display, otherwise unnecessary
        let lastSampleId = null;
        let lastActivations = null;
        // build a list of IDs to sample in this batch. we could do build this randomly
        const sampleIds = [];
        for (let innerSampleIdx = 0; (innerSampleIdx < batchSize)
            && (batchStartId + innerSampleIdx < mnistData.numberOfSamples); ++innerSampleIdx) {
            sampleIds[innerSampleIdx] = batchStartId + innerSampleIdx;
        }
        // loop until the error is below ERROR_THRESHOLD, or MAX_ITERATIONS reached
        let error = 1.0;
        for (let iterIdx = 0; (iterIdx < MAX_ITERATIONS) && (error > ERROR_THRESHOLD); ++iterIdx) {
            // accumulates the error of each sample
            let errorSum = 0.0;
            // for each sample
            for (let sampleIdx = 0; sampleIdx < sampleIds.length; ++sampleIdx) {
                // get the input activations from MNIST data
                const inputActivations = getInputActivations(mnistData, sampleIds[sampleIdx]);
                // get the expected output from MNIST label data
                const expectedOutputs = getOutputActivations(mnistData, sampleIds[sampleIdx]);
                // train this sample
                const { networkError, networkActivations } = trainPattern(network, inputActivations, expectedOutputs);
                // accumulate error
                errorSum += networkError;
                // store last sample id and activations
                lastSampleId = sampleIds[sampleIdx];
                lastActivations = networkActivations;
            }
            // update calculated error
            error = errorSum / sampleIds.length;
        }
        // return sample and activation data for the gui
        return { lastSampleId, lastActivations };
    }
    exports.trainBatch = trainBatch;
    // loadMNISTData
    // --
    // Loads the data and label files, using the HTML5 fetch api.
    // This returns a Promise to handle control flow of the asynchronous file load.
    // trainingSet is 'train' or 't10k'
    const MNIST_DIR = "./mnist";
    function loadMNISTData(trainingSet) {
        // load the label data as an ArrayBuffer
        return fetch(`${MNIST_DIR}/${trainingSet}-labels.idx1-ubyte`).then((r) => r.arrayBuffer()).then((labelBuffer) => {
            // we use DataView type to read binary data in JavaScript
            const fullLabelData = new DataView(labelBuffer, 0);
            // first value of header is magic number, used for linux filetypes
            if (fullLabelData.getUint32(0, false) !== 2049) {
                throw new Error("labels magic number mismatch");
            }
            // fetch the image data as an ArrayBuffer
            return fetch(`${MNIST_DIR}/${trainingSet}-images.idx3-ubyte`).then((r) => r.arrayBuffer()).then((imagesBuffer) => {
                // load image data as DataView
                const fullImagesData = new DataView(imagesBuffer, 0);
                // check magic number again
                if (fullImagesData.getUint32(0, false) !== 2051) {
                    throw new Error("images magic number mismatch");
                }
                // header data, includes numberOfSamples,
                // imageWidth (28), and imageHeight (28)
                const numberOfSamples = fullImagesData.getUint32(4, false);
                const imageHeight = fullImagesData.getUint32(8, false);
                const imageWidth = fullImagesData.getUint32(12, false);
                // Our MNISTData!
                return {
                    numberOfSamples, imageWidth, imageHeight,
                    labelsBuffer: new DataView(labelBuffer, 8),
                    imagesBuffer: new DataView(imagesBuffer, 16)
                };
            });
        });
    }
    exports.loadMNISTData = loadMNISTData;
    // getSampleLabel
    // --
    // Gets the expected numerical digit of the sample
    function getImageLabel({ labelsBuffer }, sampleId) {
        return labelsBuffer.getUint8(sampleId);
    }
    exports.getImageLabel = getImageLabel;
    // getInputActivations
    // --
    // Gets input activations of an image sample as a Float32Array.
    // Activations are scaled to be 0.0 to 1.0.
    function getInputActivations({ imageWidth, imageHeight, imagesBuffer }, sampleId) {
        // size of input
        const sizeOfInput = imageWidth * imageHeight;
        // our result array
        const inputActivations = new Float32Array(sizeOfInput);
        // for each pixel in the image
        for (let r = 0; r < imageHeight; ++r) {
            for (let c = 0; c < imageWidth; ++c) {
                inputActivations[r * imageWidth + c] =
                    imagesBuffer.getUint8(sampleId * sizeOfInput + r * imageWidth + c) / 256.0;
            }
        }
        return inputActivations;
    }
    exports.getInputActivations = getInputActivations;
    // getOutputActivations
    // --
    // Gets output activations of an image label as one-hot Float32Array
    function getOutputActivations(mnistData, sampleId) {
        const l = getImageLabel(mnistData, sampleId);
        const oa = new Float32Array(10);
        oa[l] = 1.0;
        return oa;
    }
    exports.getOutputActivations = getOutputActivations;
});
define("figures", ["require", "exports", "preact", "perceptron-mnist-demo"], function (require, exports, preact_1, perceptron_mnist_demo_1) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    //
    // This file contains all the figures for the post.
    // The first few are faked with SVG, but DemoNetworkTrainer
    // loads and trains the code in the blog post.
    //
    // PerceptronFigure
    // --
    // SVG Diagram of a perceptron
    class PerceptronFigure extends preact_1.Component {
        render() {
            return preact_1.h("svg", { width: 320, height: 100, viewBox: '0 0 320 100' },
                preact_1.h("circle", { r: 15, cx: 145, cy: 50, fill: 'none', stroke: 'black' }),
                preact_1.h("line", { x1: 50, y1: 30, x2: 130, y2: 50, fill: 'none', stroke: '#aaaa00' }),
                preact_1.h("line", { x1: 50, y1: 40, x2: 130, y2: 50, fill: 'none', stroke: '#aaaa00' }),
                preact_1.h("line", { x1: 50, y1: 50, x2: 130, y2: 50, fill: 'none', stroke: '#aaaa00' }),
                preact_1.h("line", { x1: 50, y1: 60, x2: 130, y2: 50, fill: 'none', stroke: '#aaaa00' }),
                preact_1.h("line", { x1: 50, y1: 70, x2: 130, y2: 50, fill: 'none', stroke: '#aaaa00' }),
                preact_1.h("text", { x: 0, y: 55, fill: "#0000cc" }, "inputs"),
                preact_1.h("text", { x: 55, y: 90, fill: "#aaaa00" }, "weights"),
                preact_1.h("text", { x: 133, y: 53, "font-size": "12", fill: "#aaaa00" }, "sum"),
                preact_1.h("line", { x1: 160, y1: 50, x2: 170, y2: 50, fill: 'none', stroke: '#99aa00' }),
                preact_1.h("text", { x: 171, y: 53, "font-size": "12", fill: "#cc0000" }, "+bias"),
                preact_1.h("line", { x1: 205, y1: 50, x2: 215, y2: 50, fill: 'none', stroke: '#aa9900' }),
                preact_1.h("circle", { r: 15, cx: 230, cy: 50, fill: 'none', stroke: 'black' }),
                preact_1.h("text", { x: 205, y: 90, fill: "#0000cc" }, "sigmoid"),
                preact_1.h("text", { x: 219, y: 59, "font-size": "32" }, "~"),
                preact_1.h("line", { x1: 245, y1: 50, x2: 260, y2: 50, fill: 'none', stroke: '#00cc00' }),
                preact_1.h("text", { x: 262, y: 55, fill: "#00cc00" }, "output"));
        }
    }
    exports.PerceptronFigure = PerceptronFigure;
    // NetworkFigure
    // --
    // SVG Diagram of a perceptron
    function MiniPerceptronFigure(props) {
        return preact_1.h("g", Object.assign({}, props),
            preact_1.h("circle", { r: 15, cx: 145, cy: 50, fill: 'none', stroke: 'black' }),
            preact_1.h("line", { x1: 50, y1: 30, x2: 130, y2: 50, fill: 'none', stroke: '#aaaa00' }),
            preact_1.h("line", { x1: 50, y1: 40, x2: 130, y2: 50, fill: 'none', stroke: '#aaaa00' }),
            preact_1.h("line", { x1: 50, y1: 50, x2: 130, y2: 50, fill: 'none', stroke: '#aaaa00' }),
            preact_1.h("line", { x1: 50, y1: 60, x2: 130, y2: 50, fill: 'none', stroke: '#aaaa00' }),
            preact_1.h("line", { x1: 50, y1: 70, x2: 130, y2: 50, fill: 'none', stroke: '#aaaa00' }),
            preact_1.h("line", { x1: 160, y1: 50, x2: 170, y2: 50, fill: 'none', stroke: '#99aa00' }),
            preact_1.h("circle", { r: 15, cx: 185, cy: 50, fill: 'none', stroke: 'black' }),
            preact_1.h("line", { x1: 200, y1: 50, x2: 215, y2: 50, fill: 'none', stroke: '#00cc00' }));
    }
    function MiniPerceptronLayer(props) {
        return preact_1.h("g", Object.assign({}, props),
            preact_1.h(MiniPerceptronFigure, { transform: "translate(0,0) scale(0.2,0.5)" }),
            preact_1.h(MiniPerceptronFigure, { transform: "translate(0,20) scale(0.2,0.5)" }),
            preact_1.h(MiniPerceptronFigure, { transform: "translate(0,40) scale(0.2,0.5)" }),
            preact_1.h(MiniPerceptronFigure, { transform: "translate(0,60) scale(0.2,0.5)" }),
            preact_1.h(MiniPerceptronFigure, { transform: "translate(0,80) scale(0.2,0.5)" }),
            preact_1.h(MiniPerceptronFigure, { transform: "translate(0,100) scale(0.2,0.5)" }));
    }
    class NetworkFigure extends preact_1.Component {
        render() {
            return preact_1.h("svg", { width: 300, height: 180, viewBox: '0 0 300 180' },
                preact_1.h("text", { x: 10, y: 95, fill: "#0000cc" }, "inputs"),
                preact_1.h("text", { x: 130, y: 15, fill: "#aaaa00" }, "layers"),
                preact_1.h(MiniPerceptronLayer, { transform: "translate(70,20)" }),
                preact_1.h(MiniPerceptronLayer, { transform: "translate(110,20)" }),
                preact_1.h(MiniPerceptronLayer, { transform: "translate(150,20)" }),
                preact_1.h(MiniPerceptronLayer, { transform: "translate(190,20)" }),
                preact_1.h("rect", { x: 116, y: 25, width: 38, height: 140, stroke: "#aaaa00", fill: "transparent" }),
                preact_1.h("text", { x: 242, y: 95, fill: "#00cc00" }, "output"));
        }
    }
    exports.NetworkFigure = NetworkFigure;
    // NetworkFigure
    // --
    // Animated svg simulating an activated perceptron
    class ActivationFigure extends preact_1.Component {
        render() {
            return preact_1.h("svg", { width: 300, height: 180, viewBox: '0 0 300 180' },
                preact_1.h("text", { x: 10, y: 55, fill: "#0000cc" }, "inputs"),
                preact_1.h("circle", { r: 10, cx: 85, cy: 30, stroke: 'none', className: 'afc-i1' }),
                preact_1.h("circle", { r: 10, cx: 85, cy: 80, stroke: 'none', className: 'afc-i2' }),
                preact_1.h("circle", { r: 15, cx: 185, cy: 50, stroke: 'none', className: 'afc-o' }),
                preact_1.h("line", { x1: 95, y1: 30, x2: 170, y2: 50, fill: 'none', stroke: '#00aa00' }),
                preact_1.h("line", { x1: 95, y1: 80, x2: 170, y2: 50, fill: 'none', stroke: '#aa0000' }),
                preact_1.h("text", { x: 110, y: 28, fill: "#00cc00" }, "+1.0"),
                preact_1.h("text", { x: 112, y: 92, fill: "#cc0000" }, "-1.0"),
                preact_1.h("text", { x: 222, y: 55, fill: "#00cc00" }, "output"));
        }
    }
    exports.ActivationFigure = ActivationFigure;
    class DemoValueCanvas extends preact_1.Component {
        constructor() {
            super(...arguments);
            // bind and draw the canvas when the element is inserted
            this.bindLayerCanvas = (el) => {
                this.el = el;
                if (el) {
                    setTimeout(this.drawLayerCanvas, 1);
                }
            };
            // draws the canvas
            this.drawLayerCanvas = () => {
                const { width, height, color, values } = this.props;
                // skip if unbound
                if (!this.el)
                    return;
                // get a 2d canvas context
                const rlCtx = this.el.getContext('2d');
                if (!rlCtx)
                    return;
                // disable antialias
                rlCtx.webkitImageSmoothingEnabled = false;
                // fill with background color
                rlCtx.fillStyle = `rgba(40,40,40,1)`;
                rlCtx.fillRect(0, 0, width, height);
                // for each layer value, draw the pixel at opacity
                if (values) {
                    for (let r = 0; r < height; ++r) {
                        for (let c = 0; c < width; ++c) {
                            const i = r * width + c;
                            const v = values[i] || 0;
                            rlCtx.fillStyle = `rgba(${color},${v})`;
                            rlCtx.fillRect(c, r, 1, 1);
                        }
                    }
                }
            };
        }
        // update on new props
        componentWillReceiveProps() {
            if (this.el) {
                setTimeout(this.drawLayerCanvas, 1);
            }
        }
        // render!
        render() {
            const { width, height, style } = this.props;
            return (preact_1.h("canvas", { className: 'demo-value-canvas', width: width, height: height, style: style, ref: this.bindLayerCanvas }));
        }
    }
    exports.DemoValueCanvas = DemoValueCanvas;
    // DemoLayerRow
    // --
    // Renders all the weight and activation canvases of a layer
    class DemoLayerRow extends preact_1.Component {
        render() {
            const { layer, layerIdx, inputActivations } = this.props;
            return (preact_1.h("div", { className: 'demo-layer-row' },
                preact_1.h("div", { className: 'demo-layer-label' },
                    "layer#",
                    layerIdx),
                preact_1.h("div", { className: 'demo-layer-row-canvases' }, layer && layer.perceptrons.map((perceptron, perceptronIdx) => {
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
                    return (preact_1.h("div", { className: 'demo-layer-perceptron-group' },
                        preact_1.h(DemoValueCanvas, { width: laSideLen, height: laSideLen, values: weightOpacities, color: '240,255,0' }),
                        preact_1.h(DemoValueCanvas, { width: laSideLen, height: laSideLen, values: weightedActivations, color: '0,255,0' })));
                }))));
        }
    }
    exports.DemoLayerRow = DemoLayerRow;
    // DemoNetworkTrainer
    // --
    // The big kahuna. Loads the data, trains the network in batches,
    // and evaluates t10k
    const BATCH_SIZE = 10;
    const BATCH_DELAY = 10; // milliseconds
    class DemoNetworkTrainer extends preact_1.Component {
        constructor() {
            super(...arguments);
            // state for this gui
            this.state = {
                loading: false,
                loaded: false,
                started: false,
                paused: false,
                trained: false,
                mnistData: null,
                network: perceptron_mnist_demo_1.CreatePerceptronNetwork(28 * 28, [10]),
                sampleId: null,
                inputActivations: null,
                activations: null,
            };
            // loads the data
            this.onLoadClick = () => {
                this.setState({ loading: true });
                perceptron_mnist_demo_1.loadMNISTData('t10k').then((mnistData) => {
                    this.setState({
                        mnistData, loaded: true,
                        sampleId: 0,
                        inputActivations: perceptron_mnist_demo_1.getInputActivations(mnistData, 0),
                    });
                });
            };
            // creates a new network object to prepare training
            this.onStartTrainClick = () => {
                const { mnistData } = this.state;
                this.setState({ started: true }, this.beginTraining);
            };
            // begins training
            this.beginTraining = () => {
                // instead of a for loop, we define a function so we can run async
                // and allow the gui to update in-between
                this.processNextBatch(0);
            };
            this.processNextBatch = (batchSampleId) => {
                const { paused, mnistData, network } = this.state;
                // stop if paused
                if (paused)
                    return;
                // while we have more sampleIds
                if (batchSampleId < mnistData.numberOfSamples) {
                    // train this batch
                    const { lastSampleId, lastActivations } = perceptron_mnist_demo_1.trainBatch(network, mnistData, batchSampleId, BATCH_SIZE);
                    // update the gui
                    this.setState({ network,
                        sampleId: lastSampleId,
                        inputActivations: perceptron_mnist_demo_1.getInputActivations(mnistData, lastSampleId),
                        activations: lastActivations,
                    });
                    // do the next batch after a short delay
                    setTimeout(() => this.processNextBatch(batchSampleId + BATCH_SIZE), BATCH_DELAY);
                }
                else {
                    // all done
                    this.setState({ trained: true });
                }
            };
            this.onTogglePauseTraining = () => {
                this.setState({ paused: !this.state.paused }, () => {
                    if (!this.state.paused)
                        this.beginTraining();
                });
            };
        }
        render() {
            const { loading, loaded, started, trained, paused, mnistData, sampleId, activations, inputActivations, network } = this.state;
            return (preact_1.h("div", { className: 'demo-trainer' },
                !loaded ? (preact_1.h("div", { className: 'demo-group', style: { marginTop: 20 } },
                    preact_1.h("div", null, "Waiting to load data"),
                    preact_1.h("button", { onClick: this.onLoadClick, disabled: loading }, "Load MNIST T10K Data (8MB)"),
                    loading ?
                        preact_1.h("div", { className: 'demo-group' }, "Loading")
                        : null)) :
                    preact_1.h("div", { className: 'demo-group' },
                        preact_1.h("div", null,
                            "Loaded ",
                            mnistData.numberOfSamples,
                            " Samples"),
                        started ?
                            preact_1.h("div", null,
                                preact_1.h("div", null,
                                    "Trained ",
                                    1 + sampleId,
                                    " Samples"),
                                preact_1.h("button", { onClick: this.onTogglePauseTraining },
                                    paused ? 'Resume' : 'Pause',
                                    " Training"))
                            :
                                preact_1.h("button", { onClick: this.onStartTrainClick }, "Begin Training")),
                preact_1.h("div", { className: 'demo-group' },
                    preact_1.h("div", { className: 'demo-layers' },
                        preact_1.h("div", { className: 'current-sample' },
                            preact_1.h("div", { className: 'demo-layer-label' }, "in"),
                            preact_1.h(DemoValueCanvas, { width: 28, height: 28, values: inputActivations, color: '0,64,255' })),
                        network && network.layers.map((layer, layerIdx) => (preact_1.h(DemoLayerRow, { layer: layer, layerIdx: layerIdx, inputActivations: (layerIdx === 0) ? inputActivations :
                                activations && activations[layerIdx - 1] }))),
                        preact_1.h("div", { className: 'outputs-sample' },
                            preact_1.h("div", { className: 'demo-layer-label' }, "out"),
                            preact_1.h(DemoValueCanvas, { width: 1, height: 10, values: activations && activations[activations.length - 1], color: '0,255,64' }))))));
        }
    }
    exports.DemoNetworkTrainer = DemoNetworkTrainer;
});
define("index", ["require", "exports", "preact", "figures"], function (require, exports, preact, Figures) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    const { Component, h } = preact;
    window.preact = preact;
    window.h = preact.h;
    // This file loads the source code and renders the blog post,
    // it's not as well documented.
    function parseCommentToTsx(text) {
        if (text.indexOf('//=') === 0) {
            const figureType = text.substring(3);
            const Figure = Figures[figureType];
            if (!Figure)
                throw new Error("figure not found " + figureType);
            return h(Figure, null);
        }
        else if (text.indexOf('//') === 0) {
            return h("div", null, text.substr(2).trim().split(' ').map((word) => ((/^\[.+\]\(.+\)$/g.test(word)) ? (() => {
                const firstCloseBracket = word.indexOf(']');
                const text = word.substring(1, firstCloseBracket);
                const url = word.substring(firstCloseBracket + 2, word.length - 1);
                return (h("span", null,
                    h("a", { href: url }, text),
                    " "));
            })() :
                h("span", null,
                    word,
                    " "))));
        }
        else {
            // empty?
            return h("div", null);
        }
    }
    exports.parseCommentToTsx = parseCommentToTsx;
    const TS_KEYWORDS_RGX = /(for)|(while)|(if)|(else)|(new)|(return)|(var)|(let)|(const)|(function)|(class)|(extends)|(export)|(import)|(interface)/g;
    function parseCodeToTsx(text) {
        // if line is comment line
        if (/ *\/\//g.test(text)) {
            return h("div", { style: { color: 'gray' } }, text);
        }
        // we replace keywords, fns, and variable names
        // we special strings for easier replacement later
        let annotatedText = text.replace(TS_KEYWORDS_RGX, (keywordText) => {
            return ':!~!:' + '$color:#00cc00~!:' + keywordText + ':!~!:';
        });
        // fn calls or defs
        annotatedText = annotatedText.replace(/([a-zA-Z_][a-zA-Z0-9_]+)\(/g, (fm, fnName) => {
            return ':!~!:' + '$color:#00ccff~!:' + fnName + ':!~!:' + '(';
        });
        return (h("div", null, annotatedText.split(':!~!:').map((word) => (word.indexOf("$color:") === 0 ?
            h("span", { style: { color: word.substring(7, 14) } }, word.substr(17))
            :
                word.length > 0 ?
                    h("span", null, word)
                    : null))));
    }
    exports.parseCodeToTsx = parseCodeToTsx;
    // Layout root preact element
    class Layout extends Component {
        constructor() {
            super(...arguments);
            this.state = {
                parsedGroups: [],
            };
            this.loadSource = () => {
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
                                        && (tg.code[tg.code.length - 1].children[0].length === 0))) {
                                    tg.code.pop();
                                }
                            }
                            parsedGroups.push({ comments: [], code: [] });
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
                                    cmts[cmts.length - 1] = h("h1", null, cmts[cmts.length - 1]);
                                }
                                skip = true;
                                // sub header market
                            }
                            else if ((line.indexOf('// --') === 0)) {
                                if (topGroup().comments.length > 0) {
                                    const cmts = topGroup().comments;
                                    cmts[cmts.length - 1] = h("h2", null, cmts[cmts.length - 1]);
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
                        }
                        else if (line.trim().length > 0) {
                            // add code to group
                            topGroup().code.push(parseCodeToTsx(line));
                            // update lastType
                            lastType = 'code';
                            // else empty
                        }
                        else {
                            // empty between code is preserved
                            if (lastType === 'code') {
                                topGroup().code.push(parseCodeToTsx(line));
                            }
                        }
                    });
                    this.setState({ parsedGroups });
                });
            };
        }
        componentDidMount() {
            this.loadSource();
        }
        render() {
            const { parsedGroups } = this.state;
            return (h("div", { className: 'blog-layout' }, parsedGroups.map((group) => (h("div", { className: 'layout-group' },
                group.comments.length > 1 ?
                    h("div", { className: 'layout-comments' }, group.comments)
                    : null,
                group.code.length > 1 ?
                    h("div", { className: 'layout-code' }, group.code)
                    : null)))));
        }
    }
    exports.Layout = Layout;
    // This is called from the HTML file to start the preact app
    function render(containerEl) {
        if (containerEl != null) {
            preact.render(h(Layout, null), containerEl, containerEl.lastElementChild || undefined);
        }
    }
    exports.render = render;
});
//# sourceMappingURL=index.js.map