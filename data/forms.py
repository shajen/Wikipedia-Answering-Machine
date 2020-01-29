from django import forms

class QuestionForm(forms.Form):
    questions = forms.CharField(label='questions', widget=forms.Textarea)
