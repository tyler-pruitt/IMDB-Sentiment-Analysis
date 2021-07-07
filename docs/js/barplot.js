var ctx = document.getElementById("barchart");

var barchart = new Chart(ctx, {
  type: 'bar',
  data: {
    labels: ['Negative', 'Positive'],
    datasets: [{
      label: 'Probability',
      data: [0.25, 0.75],
      backgroundColor: [
        'rgba(255, 80, 0, 0.8)',
        'rgba(4, 200, 6, 0.8)'
      ],
      borderColor: [
        'rgba(255, 80, 0, 1)',
        'rgba(4, 200, 6, 1)'
      ],
      borderWidth: 1
    }]
  },
  options: {
    responsive: false,
    scales: {
      xAxes: [{
        ticks: {
          maxRotation: 90,
          minRotation: 80
        },
          gridLines: {
          offsetGridLines: true // à rajouter
        }
      },
      {
        position: "top",
        ticks: {
          maxRotation: 90,
          minRotation: 80
        },
        gridLines: {
          offsetGridLines: true // et matcher pareil ici
        }
      }],
      yAxes: [{
        ticks: {
          beginAtZero: true
        }
      }]
    }
  }
});

function addData(chart, label, data) {
    chart.data.labels.push(label);
    chart.data.datasets.forEach((dataset) => {
        dataset.data.push(data);
    });
    chart.update();
}

function removeData(chart) {
    chart.data.labels.pop();
    chart.data.datasets.forEach((dataset) => {
        dataset.data.pop();
    });
    chart.update();
}
