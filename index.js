'use strict';

const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');
require('@tensorflow/tfjs-node-gpu');

(async () => {
	
	const model = tf.sequential();

	model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
	model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });

	const xs = tf.tensor1d([3.2, 4.4, 5.5]);
	const ys = tf.tensor1d([1.6, 2.7, 3.5]);

	model.fit(xs, ys, { epochs: 1024 }).then(function() {
		model.predict(tf.tensor2d([5], [1, 1])).print();
	});

	/*
	const model = tf.sequential();

	model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

	// Prepare the model for training: Specify the loss and the optimizer.
	model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });

	// Generate some synthetic data for training.
	const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
	const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);

	// Train the model using the data.
	model.fit(xs, ys).then(function(){
		// Use the model to do inference on a data point the model hasn't seen before:
		// Open the browser devtools to see the output
		model.predict(tf.tensor2d([5], [1, 1])).print();

		model.save('file://' + __dirname + '/model');

	});
	*/

	/*
	const model = tf.sequential();
	//const input = tf.input({shape: [5]});
	model.add(tf.layers.dense({ units: 1, inputShape: [10], activation: 'sigmoid' }));
	model.save('file://' + __dirname + '/model');
	*/

	/*
	const x = tf.input({ shape: [32] });
	const y = tf.layers.dense({ units: 3, activation: 'softmax' }).apply(x);
	const model = tf.model({ inputs: x, outputs: y });
	model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });
	//await model.save('file://' + __dirname + '/model');
	*/

	//console.dir(model.toJSON(null, false));

})();

/*
const r = function(min, max) {
	min = min || 0;
	max = max || 1;
	let rand = min + Math.floor(Math.random() * (max + 1 - min));
	return rand;
}

async function run() {
	// Create a simple model.
	const model = tf.sequential();
	model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

	// Prepare the model for training: Specify the loss and the optimizer.
	model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });

	let data1 = [];
	let data2 = [];
	let size = [16, 1];
	for (let i = 0; i < (size[0] * size[1]); i++) {
		let x = r(0, 100);
		data1.push(x);
		data2.push(x*0.5 - 1);
	}

	// Generate some synthetic data for training. (y = 2x - 1)
	const xs = tf.tensor2d(data1, size);
	const ys = tf.tensor2d(data2, size);

	// Train the model using the data.
	await model.fit(xs, ys, { epochs: 1024 });

	console.dir(model.predict(tf.tensor2d([size[0]], [1, 1])).dataSync());
}

run();
*/



/*
const model = tf.sequential();
model.add(tf.layers.inputLayer({ inputShape: [1024] }));
model.add(tf.layers.dense({ units: 1024, activation: 'relu' }));
model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
model.compile({
	optimizer: tf.train.adam(1e-6),
	loss: tf.losses.sigmoidCrossEntropy,
	metrics: ['accuracy']
});
*/

/*
const r = function(min, max) {
	min = min || 0;
	max = max || 1;
	let rand = min + Math.floor(Math.random() * (max + 1 - min));
	return rand;
}

let data = [];
let size = [2, 5];
for (let i = 0; i < (size[0] * size[1]); i++) {
	data.push(r(0, 100));
}

tf.tensor(data, size).print();
*/

/*
// Train a simple model:
const model = tf.sequential();
model.add(tf.layers.dense({ units: 100, activation: 'relu', inputShape: [10] }));
model.add(tf.layers.dense({ units: 1, activation: 'linear' }));
model.compile({ optimizer: 'sgd', loss: 'meanSquaredError' });

const xs = tf.randomNormal([100, 10]);
const ys = tf.randomNormal([100, 1]);

model.fit(xs, ys, {
	epochs: 100,
	callbacks: {
		onEpochEnd: async (epoch, log) => {
			console.log(`Epoch ${epoch}: loss = ${log.loss}`);
		}
	}
});
*/


/*

// Define a model for linear regression.
const model = tf.sequential();
model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

// Prepare the model for training: Specify the loss and the optimizer.
model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });

// Generate some synthetic data for training.
const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);

// Train the model using the data.
model.fit(xs, ys, { epochs: 25 }).then(() => {
	// Use the model to do inference on a data point the model hasn't seen before:
	model.predict(tf.tensor2d([5], [1, 1])).print();
});

*/