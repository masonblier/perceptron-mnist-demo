// perceptron-mnist-demo
// ==

// AKA Solving(ish) MNIST from scratch in your browser!

// By [masonblier](https://github.com/masonblier) 2019-01-05

​

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

function sigmoid(n: number): number {
  return 1 / (1 + Math.exp(-n));
}

// mse
// --
// mean squared error for returning a positive error-value between 0.0 and 1.0.
// It takes a "Float32Array", which is JavaScript-language for float[]

function mse(errors: Float32Array) {
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


// The Perceptron
// ==
// This is the fundamental unit of our network. It's one of the simplest model
// in machine learning, but highly powerful, and perhaps even intuitive.

// A perceptron uses linear transformations to represent a selective neuron.
// The dendrites reach to the lower layer, either input, or another perceptron
// layer, and pull each value through a weight. Then the whole activation is shifted
// by the "bias" value. The result is our perceptron's output, the weighted sum
// of its activations.

​

// So, data wise, we need a single bias, and a weight for each item in the
// lower layer.

//=PerceptronFigure


interface Perceptron {
  bias: number;
  weights: Float32Array;
}


// CreatePerception
// --
// Creates a single perception. We need the size of the lower layer to know
// how many weights to store.

function CreatePerceptron(sizeOfLowerLayer: number): Perceptron {
  // One weight for each item of the lower layer. We need to intiate our
  // perceptron randomly to help it converge.
  const weights = new Float32Array(sizeOfLowerLayer);
  for (let weightIdx = 0; weightIdx < sizeOfLowerLayer; ++weightIdx) {
    weights[weightIdx] = randomWeight();
  }

  // A random bias
  const bias = randomWeight();

  // Our Perceptron!
  return {bias, weights} as Perceptron;
}


// The Network
// ==
// Now we can build a neural network of perceptrons. A network is a stack
// of layers, and a layer is a row of perceptrons.

interface PerceptronNetwork {
  layers: PerceptronLayer[];
}

interface PerceptronLayer {
  perceptrons: Perceptron[];
}

// To create the network, we need to prepopulate the layers. We also
// need to know the size of the input layer, to know how many weights
// each perceptron of the first hidden layer (or output layer if only one)
// needs to have.

//=NetworkFigure

export function CreatePerceptronNetwork(
  sizeOfInput: number,
  otherLayerSizes: number[]
): PerceptronNetwork {

  // we need at least one layer for output
  if (otherLayerSizes.length < 1) {
    throw new Error("otherLayerSizes requires at least one layer");
  }

  // we build each layer from the sizes list
  const layers = otherLayerSizes.map((layerSize: number, layerIdx: number) => {
    // each perceptron needs one weight for each item in the lower layer.
    // but if we are the lowest layer, then we use the input size
    const sizeOfLowerLayer = (layerIdx === 0) ? sizeOfInput : otherLayerSizes[layerIdx - 1];

    // build and init our array of perceptrons
    const perceptrons = new Array(layerSize);
    for (let perceptronIdx = 0; perceptronIdx < layerSize; ++perceptronIdx) {
      perceptrons[perceptronIdx] = CreatePerceptron(sizeOfLowerLayer);
    }

    // Our PerceptronLayer!
    return {perceptrons} as PerceptronLayer;
  });

  // Our PerceptronNetwork!
  return {layers} as PerceptronNetwork;
}


// Activate!
// ==
// Now that we have the structure of the network, we can run inputs through it
// and get outputs. Of course, we will need to train it before it can get things
// correct, but it's necessary to use these functions in our training functions,
// and these are easier to understand, so let's write them first.

//=ActivationFigure

​

// getLayerActivations
// --
// At the core of it all is propagating the activation values through each layer.
// A layer gets its activations from each dendrite, reaching down, and pulling
// the value up, scaling it by the weight, and feeding our sum. Then our sum
// is shifted by the bias, pushing us up, or down into the dark, never to
// activate again.
// Takes the layer and activations to feed as inputs, and returns the outputs
// activations of the layer.
function getLayerActivations(
  layer: PerceptronLayer,
  inputActivations: Float32Array
): Float32Array {

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

// getNetworkActivations
// --
// This function returns all the activations of each layer, which is useful
// for training. If you just need the output, the last activationLayer is it.

// The figure below shows what an activated perceptron with
// two inputs might look like. The lower input has a negative weight,
// so it deactivates the output!


type LayerActivations = Float32Array;
type NetworkActivations = LayerActivations[];

export function getNetworkActivations(
  network: PerceptronNetwork,
  inputActivations: Float32Array
): NetworkActivations {

  // container array to hold each layer of activations
  const networkActivations = new Array(network.layers.length) as NetworkActivations;

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
function trainPattern(
  network: PerceptronNetwork,
  inputActivations: Float32Array,
  expectedOutputs: Float32Array
): {networkError: number, networkActivations: NetworkActivations} {

  // activate all current layers with the input
  const networkActivations = getNetworkActivations(network, inputActivations);

  // our actualOutputs is the top layer of activations
  const actualOutputs = networkActivations[networkActivations.length - 1];

  // calculate delta and layer error values
  const {deltas, layerErrors} = calculateDeltasAndErrors(network, actualOutputs, expectedOutputs);

  // adjust network weights!
  // we do this in-place because creating new float[]s is expensive
  adjustWeights(network, deltas, inputActivations, networkActivations);

  // return the mean squared error of the output layer
  return {networkError: mse(layerErrors[layerErrors.length - 1]), networkActivations};
}

// calculateDeltasAndErrors
// --
// This function calculates the "delta" values of each
// perceptron, which is used for adjusting weights below

type Deltas = Float32Array[];

function calculateDeltasAndErrors(
  network: PerceptronNetwork,
  actualOutputs: Float32Array,
  expectedOutputs: Float32Array
): {deltas: Deltas, layerErrors: Float32Array[]} {

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
      } else {
        // Otherwise, calculate the error by deriving it. We do this by tracing
        // each weight to the upper layer perceptrons, and using the chain rule on
        // the activation equation in getLayerActivations. Everything is linear,
        // so the derivative is just a scalar multiplication.
        // I don't understand it perfectly yet, but 3blue1brown has a good video.
        const deltasFromUpperLayer = deltas[layerIdx + 1];
        for (let upperPerceptronIdx = 0; upperPerceptronIdx < deltasFromUpperLayer.length
        ; ++upperPerceptronIdx) {
          // get the upper layer perceptron that this delta corresponds to
          const upperPerceptron = network.layers[layerIdx + 1].perceptrons[upperPerceptronIdx];
          // use the weight that connects the upper perceptron to this one
          const weightFromDendrite = upperPerceptron.weights[perceptronIdx]
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

  return {deltas, layerErrors};
}


// adjustWeights
// --
// And finally, we can update the weights
//

// learning rate controls how strongly the weights are influenced each training.
// this is one of the hyperparameters of our network, feel free to play with it.
const LEARNING_RATE = 0.3;

function adjustWeights(
  network: PerceptronNetwork,
  deltas: Deltas,
  inputActivations: Float32Array,
  networkActivations: NetworkActivations
) {

  // for each layer
  for (let layerIdx = 0; layerIdx < network.layers.length; ++layerIdx) {
    // get the array of perceptrons for this layer
    const layerPerceptrons =  network.layers[layerIdx].perceptrons;

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

export function trainBatch(
  network: PerceptronNetwork,
  mnistData: MNISTData,
  batchStartId: number,
  batchSize: number
) {
  // store lastSampleId and lastActivations for GUI display, otherwise unnecessary
  let lastSampleId = null;
  let lastActivations = null;

  // build a list of IDs to sample in this batch. we could do build this randomly
  const sampleIds = [];
  for (let innerSampleIdx = 0; (innerSampleIdx < batchSize)
    && (batchStartId + innerSampleIdx < mnistData.numberOfSamples)
  ; ++innerSampleIdx) {
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
      const {networkError, networkActivations} =
        trainPattern(network, inputActivations, expectedOutputs);

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
  return {lastSampleId,lastActivations}
}


// We can see the training in action in the figure below. The GUI calls
// trainBatch after a setTimeout to give the GUI time to refresh, you
// can find this code in figures.tsx

//=DemoNetworkTrainer


// The MNIST data
// ==
// The MNIST set is a set of numerical digits from [here](http://yann.lecun.com/exdb/mnist/)
// Importantly, they are binary files, with a short header, and the data stored
// as integers from 0 to 256 (labels from 0 to 10). We convert these to Float32Arrays for our network.
// We store this data as DataView for later access via getInputActivations and getOutputActivations

interface MNISTData {
  numberOfSamples: number;
  imageWidth: number;
  imageHeight: number;
  labelsBuffer: DataView;
  imagesBuffer: DataView;
}

// loadMNISTData
// --
// Loads the data and label files, using the HTML5 fetch api.
// This returns a Promise to handle control flow of the asynchronous file load.
// trainingSet is 'train' or 't10k'

const MNIST_DIR = "./mnist";

export function loadMNISTData(trainingSet): Promise<MNISTData> {
  // load the label data as an ArrayBuffer
  return fetch(
    `${MNIST_DIR}/${trainingSet}-labels.idx1-ubyte`
  ).then((r) => r.arrayBuffer()).then((labelBuffer: ArrayBuffer) => {

    // we use DataView type to read binary data in JavaScript
    const fullLabelData = new DataView(labelBuffer, 0);
    // first value of header is magic number, used for linux filetypes
    if (fullLabelData.getUint32(0, false) !== 2049) {
      throw new Error("labels magic number mismatch");
    }

    // fetch the image data as an ArrayBuffer
    return fetch(
      `${MNIST_DIR}/${trainingSet}-images.idx3-ubyte`
    ).then((r) => r.arrayBuffer()).then((imagesBuffer: ArrayBuffer) => {
      // load image data as DataView
      const fullImagesData = new DataView(imagesBuffer, 0);
      // check magic number again
      if (fullImagesData.getUint32(0, false) !== 2051) {
        throw new Error("images magic number mismatch")
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
      } as MNISTData;
    });
  });
}


// getSampleLabel
// --
// Gets the expected numerical digit of the sample
export function getImageLabel({labelsBuffer}: MNISTData, sampleId: number) {
  return labelsBuffer.getUint8(sampleId);
}

// getInputActivations
// --
// Gets input activations of an image sample as a Float32Array.
// Activations are scaled to be 0.0 to 1.0.
export function getInputActivations(
  {imageWidth,imageHeight,imagesBuffer}: MNISTData,
  sampleId: number
): Float32Array {
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

// getOutputActivations
// --
// Gets output activations of an image label as one-hot Float32Array
export function getOutputActivations(mnistData: MNISTData, sampleId: number): Float32Array {
  const l = getImageLabel(mnistData, sampleId);
  const oa = new Float32Array(10);
  oa[l] = 1.0;
  return oa;
}
