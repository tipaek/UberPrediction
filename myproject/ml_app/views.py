# ml_app/views.py
from django.shortcuts import render
from .forms import PredictionForm
from .utils import predict

def predict_view(request):
    result = None
    if request.method == "POST":
        form = PredictionForm(request.POST)
        if form.is_valid():
            # Extract the data from the form
            feature1 = form.cleaned_data['feature1']
            feature2 = form.cleaned_data['feature2']
            # Create your input_data list (ensure order matches model expectation)
            input_data = [feature1, feature2]
            # Get prediction
            result = predict(input_data)
    else:
        form = PredictionForm()
    
    context = {
        'form': form,
        'result': result,
    }
    return render(request, 'ml_app/predict.html', context)
