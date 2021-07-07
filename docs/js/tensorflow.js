// Determine if the user is using a mobile device or not
var isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);

// Set up mobile settings
if (isMobile) {
  $('#sentiment').css({'font-size': '100px'});
  $('#predictButton').css({'font-size': '40px'});
  $('#clearButton').css({'font-size': '40px'});
} else {
  $('#sentiment').css({'font-size': '120px'});
  $('#predictButton').css({'font-size': '20px'});
  $('#clearButton').css({'font-size': '20px'});
}

// Predict button
$('#predictButton').click(function(){
  // Insert loading spinner into number slot
  $('#sentiment').html('<img id="spinner" src="spinner.gif"/>');
  
  // Retrieve image data
  var img = new Image();
  img.onload = function() {
    context.drawImage(img, 0, 0, 64, 64);
    data = context.getImageData(0, 0, 64, 64).data;
    var input = [];

    for(var i = 0; i < data.length; i += 4) {
      input.push(data[i + 2] / 255);
    }

    // Model input data
    console.log("Model Input:");
    console.log(input);

    // Predict character given image data
    predict(input);
  };
  img.src = canvas.toDataURL('image/png');
});

// Load the model
tf.loadLayersModel('model/model.json').then(function(model) {
  window.model = model;
  console.log("Model Loaded Successfully.");
});

// Prediction function
var predict = function(input) {
  if (window.model) {
    // If the model is loaded, make a prediction with reshaped input
    window.model.predict([tf.tensor(input).reshape([1, 64, 64])]).array().then(function(output){
      // Process the data
      output = output[0];

      // The processed output from the model
      console.log("Model Output:");
      console.log(output);

      // Determine the highest score's index
      var negScore = ( (-1 * ( (output - 0.5) / 0.5) ) + 1) / 2;
      var posScore = ( ( (output - 0.5) / 0.5) + 1) / 2;

      var scores = [negScore, posScore];

      if (output >= 0.5) {
        // Prediction is more positive than negative
        var predictedIndex = 1;
      } else {
        // Prediction is more negative than positive
        var predictedIndex = 0;
      }

      var labels = ['Negative', 'Positive'];

      // Determine the predicted character and output to website and console
      var predictedSentiment = labels[predictedIndex];
      $('#sentiment').html(predictedSentiment);
      console.log("Predicted Sentiment: " + predictedSentiment);

      var probability = scores[predictedIndex] * 100;

      var probabilityDisplay = probability.toString() + "%";
      console.log("Probability: " + probabilityDisplay);

      // Round probability to three decimals
      const roundedProbability = Math.round(probability * 1000) / 1000;
      var roundedProbabilityDisplay = roundedProbability.toString() + "%";
      $('#probability').html(roundedProbabilityDisplay);

      // Update bar plot with data
      // First remove previous data
      for (var i=0;i<2;i+=1) {
        removeData(barchart);
      }

      // Add new data
      for (var i=0;i<2;i+=1) {
        addData(barchart, labels[i], scores[i]);
      }
    });
  } else {
    // The model takes a bit to load, if we are too fast, wait
    setTimeout(function(){predict(input)}, 50);
  }
}

// Clear button
$('#clearButton').click(function(){
  $('#sentiment').html('');
  $('#probability').html('');

  // Remove existing data in bar chart
  for (var i=0;i<2;i+=1) {
    removeData(barchart);
  }
  
  // Insert 0's for data in bar chart (i.e. reset the bar chart)
  var labels = ['Negative', 'Positive'];

  for (var i=0;i<2;i+=1) {
    addData(barchart, labels[i], 0);
  }
});
