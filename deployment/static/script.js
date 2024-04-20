document.getElementById('prediction-form').addEventListener('submit', function(e) {
    e.preventDefault();

    // Create FormData object
    var data = new FormData(this);

    // Collect checkbox data
    var checkboxes = document.querySelectorAll('input[type=checkbox]');
    for (var i = 0; i < checkboxes.length; i++) {
        var checkbox = checkboxes[i];
        var value = checkbox.checked ? 1 : 0;
        data.set(checkbox.name, value.toString());
    }

    // Add selected model to form data
    var modelSelect = document.getElementById('model');
    data.set('model', modelSelect.value);

    // Send POST request
    fetch('/', {
        method: 'POST',
        body: data
    })
    .then(response => response.json())
    .then(data => {
        var modelName = modelSelect.options[modelSelect.selectedIndex].text;
        var result = 'Prediction (using ' + modelName + '): ' + data.result;
        var probabilities = 'Probability of Not-Fatal: ' + data.probabilities[1].toFixed(2) + ', Probability of Fatal: ' + data.probabilities[0].toFixed(2);
        window.alert(result + '\n' + probabilities);
    })
    .catch(error => console.error('Error:', error));

});

// Initialize the map
var map = L.map('map').setView([43.6532, -79.3832], 12);

// Add a tile layer from OpenStreetMap
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
}).addTo(map);

// Add a draggable marker to the map
var marker = L.marker([43.6532, -79.3832], {draggable: true}).addTo(map);

// Update the longitude and latitude fields when the marker is dragged
marker.on('dragend', function(event) {
    var position = marker.getLatLng();
    document.getElementById('longitude').value = position.lng;
    document.getElementById('latitude').value = position.lat;
});
