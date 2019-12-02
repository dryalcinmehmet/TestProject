from django import forms
from django.core.validators import MinValueValidator

from django import forms

class ContactForm(forms.Form):
    subject = forms.CharField(max_length=100)
    message = forms.CharField(widget=forms.Textarea)
    sender = forms.EmailField()
    cc_myself = forms.BooleanField(required=False)

class GenerateRandomUserForm(forms.Form):
    total_user = forms.IntegerField(
        label='Number of users',
        required=True,
        validators=[
            MinValueValidator(10)
        ]
    )