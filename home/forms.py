from django import forms

class ReviewForm(forms.Form):
    review = forms.CharField(label='Enter your review:', widget=forms.TextInput())
