document.querySelector('form').addEventListener('submit', function(event) {
    event.preventDefault();
    const form = event.target;
    const formData = new FormData(form);

    // Add default values for unchecked checkboxes
    const checkboxes = form.querySelectorAll('input[type="checkbox"]');
    checkboxes.forEach(function(checkbox) {
        if (!checkbox.checked) {
            formData.append(checkbox.name, '0');
        }
    });

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // Handle prediction result
        console.log(data);
    })
    .catch(error => {
        console.error('Error:', error);
    });
});
