document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const uploadArea = document.getElementById('upload-container');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const uploadPrompt = document.getElementById('upload-prompt');
    const removeImageBtn = document.getElementById('remove-image');
    const detectButton = document.getElementById('detect-button');
    const detectionResults = document.getElementById('detection-results');
    const errorToast = document.getElementById('error-toast');
    const errorMessage = document.querySelector('.error-message');
    const closeError = document.querySelector('.close-error');
    const previewImage = document.getElementById('preview-image');
    const zoomInBtn = document.getElementById('zoom-in');
    const zoomOutBtn = document.getElementById('zoom-out');
    const loadingOverlay = document.getElementById('loading-overlay');

    let currentFile = null;
    let isProcessing = false;

    // Image manipulation variables
    let currentZoom = 0.85;
    let positionX = 0;
    let positionY = 0;
    let startPosX = 0;
    let startPosY = 0;
    let isDragging = false;

    // Create a wrapper for the image to better handle transformations
    function setupImageWrapper() {
        // Remove any existing wrapper first
        const existingWrapper = document.querySelector('.image-wrapper');
        if (existingWrapper) {
            // Move the image back to the container
            while (existingWrapper.firstChild) {
                previewContainer.appendChild(existingWrapper.firstChild);
            }
            existingWrapper.remove();
        }
        
        // Create new wrapper
        const wrapper = document.createElement('div');
        wrapper.className = 'image-wrapper';
        
        // Move the image into the wrapper
        if (previewImage && previewImage.parentNode) {
            previewImage.parentNode.insertBefore(wrapper, previewImage);
            wrapper.appendChild(previewImage);
        }
        
        return wrapper;
    }

    // Image dragging and zooming functions
    function updateImageTransform() {
        const wrapper = document.querySelector('.image-wrapper');
        if (wrapper) {
            wrapper.style.transform = `scale(${currentZoom}) translate(${positionX}px, ${positionY}px)`;
        }
    }
    
    function startDrag(e) {
        // Don't start drag if we clicked a button
        if (e.target.tagName === 'BUTTON' || e.target.closest('button')) {
            return;
        }
        
        e.preventDefault();
        isDragging = true;
        
        // Get the starting position
        if (e.type === 'touchstart') {
            startPosX = e.touches[0].clientX;
            startPosY = e.touches[0].clientY;
        } else {
            startPosX = e.clientX;
            startPosY = e.clientY;
        }
        
        // Change cursor
        if (previewContainer) {
            previewContainer.style.cursor = 'grabbing';
        }
        
        // Add move and end events
        document.addEventListener('mousemove', onDrag);
        document.addEventListener('touchmove', onDrag, { passive: false });
        document.addEventListener('mouseup', endDrag);
        document.addEventListener('touchend', endDrag);
    }
    
    function onDrag(e) {
        if (!isDragging) return;
        e.preventDefault();
        
        let currentX, currentY;
        if (e.type === 'touchmove') {
            currentX = e.touches[0].clientX;
            currentY = e.touches[0].clientY;
        } else {
            currentX = e.clientX;
            currentY = e.clientY;
        }
        
        // Calculate the new position
        const deltaX = (currentX - startPosX) / currentZoom;
        const deltaY = (currentY - startPosY) / currentZoom;
        
        positionX += deltaX;
        positionY += deltaY;
        
        // Update the transform
        updateImageTransform();
        
        // Update start position for next move
        startPosX = currentX;
        startPosY = currentY;
    }
    
    function endDrag() {
        isDragging = false;
        if (previewContainer) {
            previewContainer.style.cursor = 'grab';
        }
        
        // Remove event listeners
        document.removeEventListener('mousemove', onDrag);
        document.removeEventListener('touchmove', onDrag);
        document.removeEventListener('mouseup', endDrag);
        document.removeEventListener('touchend', endDrag);
    }
    
    function setupImageControls() {
        const wrapper = document.querySelector('.image-wrapper');
        if (!wrapper || !previewImage || !zoomInBtn || !zoomOutBtn) return;
        
        // Reset variables
        currentZoom = 0.85;
        positionX = 0;
        positionY = 0;
        
        // Initialize the transform
        updateImageTransform();
        
        // Zoom controls
        zoomInBtn.addEventListener('click', function(e) {
            e.stopPropagation();
            if (currentZoom < 3.0) {
                currentZoom += 0.15;
                updateImageTransform();
            }
        });
        
        zoomOutBtn.addEventListener('click', function(e) {
            e.stopPropagation();
            if (currentZoom > 0.3) {
                currentZoom -= 0.15;
                updateImageTransform();
            }
        });
        
        // Dragging
        wrapper.addEventListener('mousedown', startDrag);
        wrapper.addEventListener('touchstart', startDrag, { passive: false });
    }

    // File handling functions
    function handleFile(file) {
        if (file && file.type.match('image.*')) {
            currentFile = file;
            
            const reader = new FileReader();
            reader.onload = function(e) {
                if (previewImage) {
                    previewImage.src = e.target.result;
                    
                    // Reset position and zoom
                    currentZoom = 0.85;
                    positionX = 0;
                    positionY = 0;
                    
                    // Setup image wrapper and controls
                    setupImageWrapper();
                    showPreview();
                    setupImageControls();
                }
            };
            reader.readAsDataURL(file);
            
            if (detectButton) {
                detectButton.disabled = false;
            }
        } else {
            showError('Please select a valid image file.');
            resetUpload();
        }
    }

    function showPreview() {
        if (uploadPrompt) {
            uploadPrompt.style.display = 'none';
        }
        if (previewContainer) {
            previewContainer.style.display = 'flex';
        }
    }

    function resetUpload() {
        currentFile = null;
        
        if (fileInput) {
            fileInput.value = '';
        }
        
        if (previewImage) {
            previewImage.src = '';
        }
        
        if (previewContainer) {
            previewContainer.style.display = 'none';
        }
        
        if (uploadPrompt) {
            uploadPrompt.style.display = 'flex';
        }
        
        if (detectButton) {
            detectButton.disabled = true;
        }
    }

    // UI functions
    function showError(message) {
        if (errorMessage && errorToast) {
            errorMessage.textContent = message;
            errorToast.style.display = 'flex';
            
            setTimeout(() => {
                errorToast.style.display = 'none';
            }, 5000);
        } else {
            console.error(message);
        }
    }

    function showLoading() {
        if (loadingOverlay) {
            loadingOverlay.style.display = 'flex';
        }
    }

    function hideLoading() {
        if (loadingOverlay) {
            loadingOverlay.style.display = 'none';
        }
    }

    // Process image function
    function processImage() {
        if (!currentFile || isProcessing) return;
        
        isProcessing = true;
        const loader = document.querySelector('.loader');
        if (loader) loader.style.display = 'inline-block';
        if (detectButton) detectButton.disabled = true;
        
        // Show loading overlay with animation
        showLoading();
        
        const formData = new FormData();
        formData.append('image', currentFile);
        
        fetch('/api/detect', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            isProcessing = false;
            if (loader) loader.style.display = 'none';
            if (detectButton) detectButton.disabled = false;
            
            // Hide loading overlay
            hideLoading();
            
            if (data.success) {
                // Replace the image in the preview container with the detection result
                if (previewImage) {
                    previewImage.src = `data:image/jpeg;base64,${data.image}`;
                    
                    // Reset position and zoom for the result image
                    currentZoom = 0.85;
                    positionX = 0;
                    positionY = 0;
                    updateImageTransform();
                }
                
                // Refresh history
                loadDetectionHistory();
            } else {
                showError(data.error || 'Failed to process image');
            }
        })
        .catch(error => {
            isProcessing = false;
            hideLoading();
            if (loader) loader.style.display = 'none';
            if (detectButton) detectButton.disabled = false;
            
            showError('Error processing image: ' + error.message);
            console.error('Error:', error);
        });
    }

    // Event listeners
    if (uploadArea) {
        uploadArea.addEventListener('click', function() {
            if (fileInput) fileInput.click();
        });
        
        uploadArea.addEventListener('dragover', function(e) {
            e.preventDefault();
            uploadArea.style.borderColor = '#4F46E5';
            uploadArea.style.backgroundColor = 'rgba(79, 70, 229, 0.03)';
        });
        
        uploadArea.addEventListener('dragleave', function() {
            uploadArea.style.borderColor = '#d1d5db';
            uploadArea.style.backgroundColor = '#f9fafb';
        });
        
        uploadArea.addEventListener('drop', function(e) {
            e.preventDefault();
            uploadArea.style.borderColor = '#d1d5db';
            uploadArea.style.backgroundColor = '#f9fafb';
            
            if (e.dataTransfer.files.length) {
                handleFile(e.dataTransfer.files[0]);
            }
        });
    }
    
    if (fileInput) {
        fileInput.addEventListener('change', function() {
            if (fileInput.files.length) {
                handleFile(fileInput.files[0]);
            }
        });
    }
    
    if (removeImageBtn) {
        removeImageBtn.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            resetUpload();
            return false;
        });
    }
    
    if (detectButton) {
        detectButton.addEventListener('click', processImage);
    }
    
    if (closeError) {
        closeError.addEventListener('click', function() {
            errorToast.style.display = 'none';
        });
    }
    
    // Initialize image controls if an image is already loaded
    if (previewImage && previewImage.src && previewImage.src !== window.location.href) {
        setupImageWrapper();
        setupImageControls();
        showPreview();
    }
    
    // Hide loading overlay on page load
    hideLoading();

    // Optionally, clear the panel on page load
    if (detectionResults) detectionResults.innerHTML = '';
});

function loadDetectionHistory() {
    fetch('/api/history')
        .then(res => res.json())
        .then(data => {
            const historyTrack = document.getElementById('history-track');
            if (!data.length) {
                historyTrack.innerHTML = '<div class="no-data">No detection history available</div>';
                return;
            }
            historyTrack.innerHTML = '';
            data.forEach(item => {
                const entry = document.createElement('div');
                entry.className = 'history-entry';
                entry.innerHTML = `
                    <strong>${item.image_filename}</strong> - 
                    Cylinders: ${item.count} 
                    <span style="color:gray;">(${new Date(item.timestamp).toLocaleString()})</span>
                `;
                historyTrack.appendChild(entry);
            });
        });
}

// Call this on page load
document.addEventListener('DOMContentLoaded', loadDetectionHistory); 