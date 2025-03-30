document.addEventListener('DOMContentLoaded', function() {
    const form = document.querySelector('form');
    const fileInput = document.querySelector('input[type="file"]');
    const submitButton = document.querySelector('.btn-primary');
    
    // Update file input label when a file is selected
    fileInput.addEventListener('change', function() {
        const fileName = this.files[0]?.name || 'Choose your dataset';
        this.nextElementSibling.textContent = fileName;
    });
    
    // Handle form submission
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Disable submit button and show loading state
        submitButton.disabled = true;
        submitButton.textContent = 'Generating Data...';
        
        try {
            const formData = new FormData(this);
            
            const response = await fetch('/generate', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            // Show success message
            const successMessage = document.createElement('div');
            successMessage.className = 'text-green-500 mt-4 text-center';
            successMessage.textContent = data.message;
            form.insertBefore(successMessage, submitButton);
            
            // Trigger download
            window.location.href = data.download_link;
            
            // Reset submit button after a short delay
            setTimeout(() => {
                submitButton.disabled = false;
                submitButton.textContent = 'Generate Data';
            }, 2000);
            
        } catch (error) {
            // Show error message
            const errorDiv = document.createElement('div');
            errorDiv.className = 'text-red-500 mt-4 text-center';
            errorDiv.textContent = error.message;
            form.insertBefore(errorDiv, submitButton);
            
            // Reset submit button
            submitButton.disabled = false;
            submitButton.textContent = 'Generate Data';
        }
    });
    
    // Add smooth scrolling for navigation links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}); 