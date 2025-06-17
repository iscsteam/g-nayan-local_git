explanation_labels = {
    0: (
        'No DR: No diabetic retinopathy detected.\n'
        'The analysis indicates a low chance of developing diabetic retinopathy if blood sugar levels are well-managed.'
    ),
    1: ( 
        'Mild NPDR (Nonproliferative Diabetic Retinopathy):\n'
        'Microaneurysms have been detected, which are small, localized dilations of blood vessels in the retina. '
        'This finding suggests a moderate chance of progression if not monitored and managed properly.'
    ),
    2: (
        'Moderate to Severe NPDR:\n'
        'The analysis shows hemorrhages, including both dot-and-blot and flame-shaped types, and prominent exudates, indicating retinal edema. '
        'These findings suggest a high chance of further progression if left untreated, requiring close monitoring and intervention.'
    ),
    3: (
        'Proliferative Diabetic Retinopathy (PDR):\n'
        'Neovascularization has been observed, characterized by the growth of new, abnormal blood vessels on the retina. '
        'Additionally, cotton wool spots may be present, indicating localized retinal ischemia. '
        'These findings carry a significant risk of vision loss and complications, necessitating urgent medical intervention.'
    )
}
CATEGORY_MAPPING = {
    0: 'No Diabetic Retinopathy',
    1: 'Mild Diabetic Retinopathy',
    2: 'Moderate to Severe Diabetic Retinopathy',
    3: 'Proliferative Diabetic Retinopathy'
}
 