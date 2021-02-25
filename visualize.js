// let tf =require ('@tensorflow/tfjs');


// possible vaues for default model are: model_cnn, model_simple, model_small_cnn
const default_model = 'model_small_cnn'

// area where user draws the digit
let canvasDrawArea;
let contextDrawArea;
const borderDrawArea = 30;

// area with converted drawed image  into 28x28 mnist size
let canvasNormalizedInput;
let contextNormalizedInput;

// areas for CNN output
let convolutionOutput;

// mouse position / state
let x;
let y;
let pressed;

// the neuronal net
let model;

// submodels. [first layer, 1st and 2nd layer, ...]
subModels = []

window.onload = init;

function init(){

	// selection of model
	document.getElementById("model_cnn").onclick = selectHandler;
	document.getElementById("model_simple").onclick = selectHandler;
	document.getElementById("model_small_cnn").onclick = selectHandler;

	// draw area
	canvasDrawArea = document.getElementById("canvasDrawArea");
	contextDrawArea = canvasDrawArea.getContext("2d");

	canvasNormalizedInput = document.getElementById("canvasNormalizedInput");
	contextNormalizedInput = canvasNormalizedInput.getContext("2d");
	contextNormalizedInput.scale(0.1, 0.1);

	pressed = false;

	canvasDrawArea.onmousedown = MouseDownHandler;
	canvasDrawArea.onmouseup = MouseUpHandler;
	canvasDrawArea.onmousemove = MouseMoveHandler;

	// border
	contextDrawArea.fillStyle = "dimgray";
	contextDrawArea.shadowBlur = 0;
    contextDrawArea.fillRect(0,0,280,borderDrawArea);
    contextDrawArea.fillRect(0,280 - borderDrawArea,280,borderDrawArea);
    contextDrawArea.fillRect(0,0,borderDrawArea,280);
    contextDrawArea.fillRect(280 - borderDrawArea,0,borderDrawArea,280);

	contextDrawArea.strokeStyle = "black";
	contextDrawArea.lineWidth = 20;
	contextDrawArea.shadowBlur = 5;
	contextDrawArea.shadowColor = "black";

	// area for output of between layers
	hiddenOutput = document.getElementById("hiddenOutput");

	document.getElementById(default_model).checked = true;
	loadModel(default_model);

}

//
// name is name of the dirctory where the file model.json is located
//
async function loadModel(folderName) {
	model = await tf.loadLayersModel('./' + folderName + '/model.json');

	// log
	// console.log('model1.summary:');
	// model1.summary()
	console.log('input: ' + eval(model.layers[0].getConfig())['batchInputShape'])
	console.log('number of layers: ' + model.layers.length)

	for (let i = 0; i < model.layers.length; i++) {
		const config = eval(model.layers[i].getConfig());

		/*
		console.log(config)
		console.log('model.layers#' + i)
		console.log('###############')
		console.log('name: ' + config['name'])
		console.log('rate:' + config['rate'])
		console.log('activation: ' + config['activation'])
		console.log('filters: ' + config['filters'])
		console.log('params: ' + model.layers[i].countParams());
		console.log('units: '+ config['units'])
		console.log('output: ' + model.layers[i].outputShape);
		console.log('useBias: ' +  config['useBias'])
		 */

	}
	// generate HTML of (full) model description into table
	showModelMetadata();

	// submodels
	for(let i = 0; i < model.layers.length - 1; i++) {
		subModels[i] = tf.sequential()
		for (let j = 0; j <= i; j++) {
			subModels[i].add(model.layers[j]);
		}
	}

}

function showModelMetadata() {

	// generate HTML of model description into table
	const thead = document.getElementById("model_description_thead");
	thead.innerHTML = "";

	for(let header of ["#", "Name der Schicht", "Aktivierungsfkt.", "Filter", "Parameter", "Ausgänge"]) {
		let th = document.createElement("th");
		th.innerHTML = header;
		thead.appendChild(th);
	}

	const tbody = document.getElementById("model_description_tbody");
	tbody.innerHTML = "";
	for(let i = 0; i < model.layers.length; i++) {

		const config = eval(model.layers[i].getConfig());

		let tr = document.createElement("tr");
		tr.className ="values";

		createAndAppendTDonTR(tr, i);
		createAndAppendTDonTR(tr, config['name']);
		createAndAppendTDonTR(tr, config['activation']);
		createAndAppendTDonTR(tr, config['filters']);
		createAndAppendTDonTR(tr, model.layers[i].countParams());
		createAndAppendTDonTR(tr, model.layers[i].outputShape);

		tbody.appendChild(tr);
	}
}

function createAndAppendTDonTR(tr, input) {
	let td = document.createElement("td");
	if (input == undefined) {
		td.innerHTML = "-";
	}
	else {
		td.innerHTML =input;
	}

	tr.appendChild(td)
}

async function predictFromDrawedImage(pixels) {

	// input matrix of pixels. 3 or4 dimensions
	//    - 1st : data
	//    - 2nd/3rd : shape
	//    - 4th (only in case of cnn) : channel
	let inputData;

	if (eval(model.layers[0].getConfig())['batchInputShape'].length == 4) {
		inputData = tf.tensor4d(pixels, [1, 28, 28, 1]);
	} else {
		inputData = tf.tensor3d(pixels, [1, 28, 28]);
	}

	// predict the digit in full net
	const preds = model.predict(inputData);

	// log
	// console.log(preds.arraySync())
	predArray = preds.arraySync();

	// update diagramm table
	for (let i = 0; i < 10; i++) {
		const outputName = document.getElementById("output_" + i);
		outputName.innerHTML = i;
		const prediction = document.getElementById("prediction_" + i);
		prediction.style.width = 2 * predArray[0][i] * 100 + "px";
		// prediction.innerText = (predArray[0][i] * 100).round(2) + "%";
	}

	showLayerResults(inputData)
}

async function showLayerResults(inputData) {
	// delete old convolutionOutput
	hiddenOutput.innerHTML = "";

	// submodels
	for (let i = 0; i < model.layers.length - 1; i++) {
		let config = eval(subModels[i].layers[i].getConfig())
		if (config['name'].startsWith('conv2d')) {
			showLayerConf2D_Maxpooling2D(inputData, 'Convolution', i, hiddenOutput)
		}
		else if (config['name'].startsWith('max_pooling2d')) {
			showLayerConf2D_Maxpooling2D(inputData, 'Max pooling', i, hiddenOutput)
		}

		else if (config['name'].startsWith('dense')) {
			showLayerDense_Flatten(inputData, 'Dense', i, hiddenOutput)
		}

		else if (config['name'].startsWith('flatten')) {
			showLayerDense_Flatten(inputData, 'Flatten', i, hiddenOutput)
		}

		else {

			// console.log("no to be shown layer#" + i)
		}
	}
}


async function showLayerConf2D_Maxpooling2D(inputData, typeName, level, htmlConvolutionOutput) {

		const predConvolutionOnly = subModels[level].predict(inputData);

		// console.log(pred1.arraySync());

		// [1][xx][xx][count_filters]  - 1 datset, xx by xx pixles and x channels
		let predArrayConvolutionOnly = predConvolutionOnly.arraySync();

		// this works only for conf2D but not max_pooling2d
		// const convolutionFilters = subModels[level].layers[level].getConfig()['filters'];
		const convolutionFilters = predArrayConvolutionOnly[0][0][0].length

		const pixels_xy = subModels[level].layers[level].outputShape[1]
		// console.log(pixels_x, pixels_y)


		// set headline in div convolutionOutput
		headline = document.createElement("h2");
		headline.innerHTML = "" + (level + 1)  + ". Zwischenschicht nach " + typeName + '(je ' + pixels_xy +'x'+ pixels_xy + 'Pixel)';

		htmlConvolutionOutput.append(headline)

		let canvasConvolutionOutput = []
		let contextConvolutionOutput = []
		let context_temp = []
		let imgData_temp = []
		let canvas_temp = []

		for (let i = 0 ; i < convolutionFilters; i++) {

			canvas_temp[i] = document.createElement("canvas");
			context_temp[i] = canvas_temp[i].getContext("2d");
			imgData_temp[i] = context_temp[i].createImageData(pixels_xy, pixels_xy);

			for(let row = 0; row < pixels_xy; row ++){
				for(let col = 0; col < pixels_xy; col ++) {
					let imgDataPixel = row * 4 * pixels_xy + col * 4;
					imgData_temp[i].data[imgDataPixel] = 0;   		// set every red pixel element to 255
					imgData_temp[i].data[imgDataPixel + 1] = 0;   	// set every green pixel element to 255
					imgData_temp[i].data[imgDataPixel + 2] = 0;   	// set every blue pixel element to 255
					imgData_temp[i].data[imgDataPixel + 3] = 255 * predArrayConvolutionOnly[0][row][col][i];
				}
			}

			context_temp[i].putImageData(imgData_temp[i],0,0);

			// canvas for convolution output 84x84px
			canvasConvolutionOutput[i] = document.createElement("canvas");
			canvasConvolutionOutput[i].id = "canvasConvolutionOutput_" + level + '_' + i;
			canvasConvolutionOutput[i].className = "canvasConvolutionOutput";
			canvasConvolutionOutput[i].height = 84;
			canvasConvolutionOutput[i].width = 84;

			htmlConvolutionOutput.appendChild(canvasConvolutionOutput[i]);

			contextConvolutionOutput[i] = canvasConvolutionOutput[i].getContext("2d");
			contextConvolutionOutput[i].scale(84.0 / pixels_xy, 84.0 / pixels_xy);
			contextConvolutionOutput[i].drawImage(canvas_temp[i], 0, 0, );

		}
}

async function showLayerDense_Flatten(inputData, typeName, level, htmlConvolutionOutput) {
	const predHidden = subModels[level].predict(inputData);

	// console.log(predHidden.arraySync());

	// [1][xx][xx][count_filters]  - 1 datset, xx by xx pixles and x channels
	let predHiddenArray = predHidden.arraySync();
	// console.log(predHiddenArray)

	neuronCount =  predHiddenArray[0].length
	// console.log(predHiddenArray[0].length)
	// console.log('min/max', Math.max.apply(null, predHiddenArray[0]), Math.min.apply(null, predHiddenArray[0]) )


	// set headline in div convolutionOutput
	let headline = document.createElement("h2");
	headline.innerHTML = "" + (level + 1)  + ". Zwischenschicht nach " + typeName + " (" + neuronCount + " Neuronen)";

	htmlConvolutionOutput.append(headline)

	let canvas_temp = document.createElement("canvas");
	let context_temp = canvas_temp.getContext("2d");

	const neuronRadius = 4
	const neuronSize = 10
	const neuronsPerRow = 90

	let canvasFlattenOutput = document.createElement("canvas");
	canvasFlattenOutput.id = "canvasFlattenOutput";
	canvasFlattenOutput.className = "canvasFlattenOutput";
	canvasFlattenOutput.height = neuronSize * (Math.floor(neuronCount / neuronsPerRow) + 1);
	canvasFlattenOutput.width = neuronsPerRow * neuronSize;
	let contextFlattenOutput = canvasFlattenOutput.getContext("2d");
	contextFlattenOutput.fillStyle = 'black';
	contextFlattenOutput.lineWidth = 1;

	const maxValue = Math.max.apply(null, predHiddenArray[0])
	const minValue = Math.min.apply(null, predHiddenArray[0])

	for (let neuronId = 0 ; neuronId<neuronCount; neuronId++) {
		x = neuronId % neuronsPerRow
		y = Math.floor(neuronId / neuronsPerRow)

		// circle
		contextFlattenOutput.beginPath()
		contextFlattenOutput.fillStyle = 'black';
		contextFlattenOutput.lineWidth = 1;
		contextFlattenOutput.arc( neuronSize / 2 + neuronSize * x, neuronSize / 2 + neuronSize * y, neuronRadius, 0, Math.PI * 2, false);
		contextFlattenOutput.stroke();

		contextFlattenOutput.beginPath();
		const value = predHiddenArray[0][neuronId];
		const normValue = 255 - (255 * (value - minValue) / (maxValue - minValue));
		contextFlattenOutput.arc( neuronSize / 2 + neuronSize * x, neuronSize / 2 + neuronSize * y, neuronRadius, 0, Math.PI * 2, false);
		contextFlattenOutput.fillStyle = 'rgb(' + normValue + ',' + normValue+ ',' + normValue +  ')';
		contextFlattenOutput.fill();

		// fill circe
		//if (predHiddenArray[0][neuronId] < 0.1) {
		//	contextFlattenOutput.fillStyle = 'black';
		//		contextFlattenOutput.lineWidth = 1;
		//		contextFlattenOutput.stroke();
		//}
		//else {
		//		const value = predHiddenArray[0][neuronId]
		//		const normValue = 255 * (value - minValue) / (maxValue - minValue)
		//
		//	contextFlattenOutput.fillStyle = 'rgb(' + normValue + ',' + normValue+ ',' + normValue +  ')';
		//	contextFlattenOutput.fill();
		//}
	}
	// contextFlattenOutput.moveTo(50, 50);
	// contextFlattenOutput.arc( 50, 50, 50, 0, Math.PI * 2, false);
	// contextFlattenOutput.stroke();

	htmlConvolutionOutput.appendChild(canvasFlattenOutput);




}


function MouseDownHandler(event){

	x = event.pageX - canvasDrawArea.offsetLeft;
	y = event.pageY - canvasDrawArea.offsetTop;
	pressed = true;
	// alert("event.pageY:" + event.pageY + " canvas_mnist.offsetTop:" + canvas_mnist.offsetTop);

}

function MouseUpHandler(event){
	pressed = false;
}


function MouseMoveHandler(event){
	// window.alert(event.pageX);
	if (pressed == true) {
		contextDrawArea.beginPath();
		contextDrawArea.moveTo(x, y);
		x = event.pageX - canvasDrawArea.offsetLeft;
		y = event.pageY - canvasDrawArea.offsetTop;

		if((x >= borderDrawArea) && (x <= 280 - borderDrawArea)  && (y > borderDrawArea)  && (y < 280 - borderDrawArea)) {
		    contextDrawArea.lineTo(x, y);
		    contextDrawArea.closePath();
		    contextDrawArea.stroke();
		}

	}
}

function ButtonResetHandler() {
	/*if (window.location.host == "")
	    debugtext.innerText = "local";
	else
	    debugtext.innerText = window.location.host;
	 */
	contextDrawArea.clearRect(0, 0, canvasDrawArea.width, canvasDrawArea.height);
	contextNormalizedInput.clearRect(0, 0, canvasDrawArea.width, canvasDrawArea.width);
	contextDrawArea.fillStyle = "dimgray";
	contextDrawArea.shadowBlur = 0;
    contextDrawArea.fillRect(0,0,280,borderDrawArea);
    contextDrawArea.fillRect(0,280 - borderDrawArea,280,borderDrawArea);
    contextDrawArea.fillRect(0,0,borderDrawArea,280);
    contextDrawArea.fillRect(280 - borderDrawArea,0,borderDrawArea,280);
    contextDrawArea.shadowBlur = 5;
    contextDrawArea.shadowColor="black"

	for(let i =0; i < 10; i++) {
		const outputName = document.getElementById("output_" + i);
		outputName.innerHTML= i;
		const prediction = document.getElementById("prediction_" + i);
		prediction.style.width = "0px";
	}

	// remove convolutionOutput
	hiddenOutput.innerHTML = "";
}

function ButtonStartHandler() {
	var pixel = []; 

	var imagedata;

	for (var i=0; i < 28*28; i++) {
		pixel[i] = 0;
	}

    // transforn drawed image ("draw") into small canvas ("data.mnist") with 28x28
    // cut border
	contextNormalizedInput.drawImage(canvasDrawArea, 0, 0);
    contextNormalizedInput.clearRect(0,0,280,borderDrawArea);
    contextNormalizedInput.clearRect(0,280 - borderDrawArea,280,borderDrawArea);
    contextNormalizedInput.clearRect(0,0,borderDrawArea,280);
    contextNormalizedInput.clearRect(280- borderDrawArea,0,borderDrawArea,280);
	imagedata = contextNormalizedInput.getImageData(0, 0, canvasNormalizedInput.width, canvasNormalizedInput.height);
	
	// debugtext.innerText = imagedata.data.length;

	var rawData = "";
	for (var i=0; i < 28*28; i++) {
		value = 0;
		for (var j=0; j < 4; j++) {
			value += imagedata.data[i*4 + j]			
		}
		pixel[i] = value / 255.0;
		rawData += value;
		rawData +=";"
	}

	//console.log(pixel);
	// console.log(rawData);

   // document.getElementById("name_header").innerHTML="Ziffer";
	//document.getElementById("prediction_header").innerHTML="Wahrscheinlichkeit";

	predictFromDrawedImage(pixel)

}

// id of selected radio buttion is foldername of model file
async function selectHandler(event){
	// console.log(event.target.id)
	await loadModel(event.target.id);
	ButtonStartHandler();
}
	
	


