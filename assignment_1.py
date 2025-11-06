import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from skfuzzy import membership as mf
import matplotlib.pyplot as plt

# ===================================
#    Initializing inputs & outputs
# ===================================
# Pollutant Concentration PM2.5 (microgram per cubic meter)
pollutant = ctrl.Antecedent(np.arange(0, 200.1, 0.1), 'pollutant')

# Temperature (degree Celsius)
temperature = ctrl.Antecedent(np.arange(0, 40.1, 0.1), 'temperature')

# Humidity (Percentage)
humidity = ctrl.Antecedent(np.arange(0, 100.1, 0.1), 'humidity')

# Urban air quality condition
air_quality = ctrl.Consequent(np.arange(0, 100.1, 0.1), 'air_quality')

# Recommendations
recommendation = ctrl.Consequent(np.arange(0, 100.1, 0.1), 'recommendation')

# ==================================
#    Defining membership function
# ==================================
# Pollutant Concentration
pollutant['low'] = mf.trimf(pollutant.universe, [0, 0, 30])
pollutant['moderate'] = mf.trapmf(pollutant.universe, [20, 40, 60, 80])
pollutant['high'] = mf.trapmf(pollutant.universe, [60, 90, 120, 150])
pollutant['very high'] = mf.trapmf(pollutant.universe, [120, 150, 200, 200])

# Temperature
temperature['cold'] = mf.trapmf(temperature.universe, [0, 0, 14, 22])
temperature['mild'] = mf.trimf(temperature.universe, [20, 25, 30])
temperature['hot'] = mf.trapmf(temperature.universe, [28, 35, 40, 40])

# Humidity
humidity['dry'] = mf.trapmf(humidity.universe, [0, 0, 30, 45])
humidity['comfortable'] = mf.trimf(humidity.universe, [35, 55, 75])
humidity['humid'] = mf.trapmf(humidity.universe, [65, 80, 100, 100])

# Urban air quality condition
air_quality['good'] = mf.trapmf(air_quality.universe, [0, 0, 25, 40])
air_quality['fair'] = mf.trimf(air_quality.universe, [30, 50, 70])
air_quality['unhealthy'] = mf.trimf(air_quality.universe, [60, 75, 90])
air_quality['hazardous'] = mf.trapmf(air_quality.universe, [85, 95, 100, 100])

# Recommendations
recommendation['safe outdoor activities'] = mf.trapmf(recommendation.universe, [0, 0, 25, 40])
recommendation['limit outdoor activities'] = mf.trimf(recommendation.universe, [30, 55, 80])
recommendation['stay indoors'] = mf.trapmf(recommendation.universe, [70, 85, 100, 100])

# ==========================
#    Defining fuzzy rules
# ==========================
# R1-R3: Neutral weather
# R1 - If pollutant is LOW & temperature is MILD & humidity is COMFORTABLE, then air quality is GOOD & recommend SAFE OUTDOOR ACTIVITIES
rule1 = ctrl.Rule(
    pollutant['low'] & temperature['mild'] & humidity['comfortable'],
    (air_quality['good'], recommendation['safe outdoor activities'])
)

# R2 - If pollutant is MODERATE & temperature is MILD & humidity is COMFORTABLE, then air quality is FAIR & recommend LIMIT OUTDOOR ACTIVITIES
rule2 = ctrl.Rule(
    pollutant['moderate'] & temperature['mild'] & humidity['comfortable'],
    (air_quality['fair'], recommendation['limit outdoor activities'])
)

# R3 - If pollutant is HIGH & temperature is MILD & humidity is COMFORTABLE, then air quality is UNHEALTHY & recommend STAY INDOORS
rule3 = ctrl.Rule(
    pollutant['high'] & temperature['mild'] & humidity['comfortable'],
    (air_quality['unhealthy'], recommendation['stay indoors'])
)

# R4-R5: Worsening 'low' pollutant with extreme weather
# R4 - If pollutant is LOW & temperature is COLD/HOT, then air quality is FAIR & recommend LIMIT OUTDOOR ACTIVITIES
rule4 = ctrl.Rule(
    pollutant['low'] & (temperature['cold'] | temperature['hot']),
    (air_quality['fair'], recommendation['limit outdoor activities'])
)

# R5 - If pollutant is LOW & humidity is DRY/HUMID, then air quality is FAIR & recommend LIMIT OUTDOOR ACTIVITIES
rule5 = ctrl.Rule(
    pollutant['low'] & (humidity['dry'] | humidity['humid']),
    (air_quality['fair'], recommendation['limit outdoor activities'])
)

# R6-R7: Worsening 'moderate' pollutant with extreme weather
# R6 - If pollutant is MODERATE & temperature is COLD/HOT, then air quality is UNHEALTHY & recommend LIMIT OUTDOOR ACTIVITIES
rule6 = ctrl.Rule(
    pollutant['moderate'] & (temperature['cold'] | temperature['hot']),
    (air_quality['unhealthy'], recommendation['limit outdoor activities'])
)

# R7 - If pollutant is MODERATE & humidity is DRY/HUMID, then air quality is UNHEALTHY & recommend LIMIT OUTDOOR ACTIVITIES
rule7 = ctrl.Rule(
    pollutant['moderate'] & (humidity['dry'] | humidity['humid']),
    (air_quality['unhealthy'], recommendation['limit outdoor activities'])
)

# R8-R9: Worsening 'high' pollutant with extreme weather
# R8 - If pollutant is HIGH & temperature is COLD/HOT, then air quality is HAZARDOUS & recommend STAY INDOORS
rule8 = ctrl.Rule(
    pollutant['high'] & (temperature['cold'] | temperature['hot']),
    (air_quality['hazardous'], recommendation['stay indoors'])
)

# R9 - If pollutant is HIGH & humidity is DRY/HUMID, then air quality is HAZARDOUS & recommend STAY INDOORS
rule9 = ctrl.Rule(
    pollutant['high'] & (humidity['dry'] | humidity['humid']),
    (air_quality['hazardous'], recommendation['stay indoors'])
)

# R10: Highest pollution level regardless of weather
# R10 - If pollutant is VERY HIGH, then air quality is HAZARDOUS & recommend STAY INDOORS
rule10 = ctrl.Rule(
    pollutant['very high'],
    (air_quality['hazardous'], recommendation['stay indoors'])
)

rules = [rule1, rule2, rule3, rule4, rule5,
         rule6, rule7, rule8, rule9, rule10]

# ========================
#    Fuzzy System Setup
# ========================
air_quality_ctrl = ctrl.ControlSystem(rules=rules)
air_quality_sim = ctrl.ControlSystemSimulation(control_system=air_quality_ctrl)

def get_memberships(var, terms, value):
    return {term: fuzz.interp_membership(var.universe, var[term].mf, value) for term in terms}

# ===================
#    Compute Tests
# ===================

# Test cases = (Description, Pollutant, Temperature, Humidity)
test_cases = [
    ("A near perfect day", 5, 23, 50),                                  # Very low pollutant, comfortable temperature & humidity
    ("A really hazardous day", 170, 25, 55),                            # Very high pollutant
    ("A moderate but slightly hot day", 38, 29, 57),                    # Moderate pollutant, slightly high temperature
    ("A slightly high pollutant and slightly dry day", 63, 27, 38),     # Slightly high pollutant, slightly dry humidity but almost comfortable
    ("A slightly moderate day with slightly bad weather", 23, 30, 67)   # Slightly moderate pollutant, also slightly high temperature and humidity
]

for test_description, test_pollutant, test_temperature, test_humidity in test_cases:

    air_quality_sim.input['pollutant'] = test_pollutant
    air_quality_sim.input['temperature'] = test_temperature
    air_quality_sim.input['humidity'] = test_humidity

    air_quality_sim.compute()

    # ==============================
    #    Final Outputs & Analysis
    # ==============================

    # Input Values & Their Memberships
    pollutant_linguistics = ['low', 'moderate', 'high', 'very high']
    temperature_linguistics = ['cold', 'mild', 'hot']
    humidity_linguistics = ['dry', 'comfortable', 'humid']
    pollutant_memberships = get_memberships(pollutant, pollutant_linguistics, test_pollutant)
    temperature_memberships = get_memberships(temperature, temperature_linguistics, test_temperature)
    humidity_memberships = get_memberships(humidity, humidity_linguistics, test_humidity)

    print("="*20)
    print("   INPUT ANALYSIS   ")
    print("="*20)
    print(f"Test situation: {test_description}")
    print(f"Pollutant Concentration: {test_pollutant} µg/m3; Temperature: {test_temperature}°C; Humidity: {test_humidity}%\n")

    for label, dict in [("Pollutant Concentration", pollutant_memberships),
                        ("Temperature", temperature_memberships),
                        ("Humidity", humidity_memberships)]:
        print(f"--- {label} Memberships ---")
        for term, value in dict.items():
            print(f"{term.upper()}: {value:.3f}")
        print()

    # Activated Rules & Their Strengths
    print("="*21)
    print("   RULE ACTIVATION   ")
    print("="*21)

    activated_rules = []
    for i, rule in enumerate(air_quality_ctrl.rules, 1):
        if hasattr(rule, 'aggregate_firing'):
            firing_strength = rule.aggregate_firing[air_quality_sim]
            if firing_strength > 0:
                activated_rules.append(i)
                print(f"Rule {i} activated [Strength: {firing_strength:.4f}]:")
                print(f"  IF {rule.antecedent}, THEN {rule.consequent}")

    if not activated_rules:
        print("No rules activated.")
    print()

    # Output Values & Their Memberships
    final_aq_value = air_quality_sim.output['air_quality']
    final_rec_value = air_quality_sim.output['recommendation']

    air_quality_linguistics = ['good', 'fair', 'unhealthy', 'hazardous']
    recommendation_linguistics = ['safe outdoor activities', 'limit outdoor activities', 'stay indoors']
    air_quality_memberships = get_memberships(air_quality, air_quality_linguistics, final_aq_value)
    recommendation_memberships = get_memberships(recommendation, recommendation_linguistics, final_rec_value)

    dominant_terms = {}
    dominant_values = {}

    print("="*21)
    print("   OUTPUT ANALYSIS   ")
    print("="*21)
    print(f"Final Air Quality Risk Score: {final_aq_value:.2f}")
    print(f"Final Recommendation Risk Score: {final_rec_value:.2f}")

    for label, dict in [("Air Quality", air_quality_memberships),
                        ("Recommendation", recommendation_memberships)]:
        print(f"--- {label} Memberships ---")
        for term, value in dict.items():
            print(f"{term.upper()}: {value:.3f}")

        dominant_term = max(dict, key=dict.get)
        dominant_value = dict[dominant_term]
        dominant_terms[label] = dominant_term
        dominant_values[label] = dominant_value

        print(f"\nDominant Category: {dominant_term.upper()} ({dominant_value:.3f})\n")

    # Visualization
    air_quality.view(sim=air_quality_sim)
    recommendation.view(sim=air_quality_sim)

    # Human Interpretable Outputs
    print("="*20)
    print("   FINAL RESULTS   ")
    print("="*20)

    final_output = f"""
Based on the current conditions:
- Pollutant concentration: {test_pollutant} µg/m3
- Temperature: {test_temperature}°C
- Humidity: {test_humidity}%

The urban air quality risk score was evaluated to be {final_aq_value:.2f}/100.00,
which falls mainly under the '{dominant_terms['Air Quality'].capitalize()}' category with a {dominant_values['Air Quality']*100:.1f}% membership.

Meanwhile, the suggested recommendation risk score was evaluated to be {final_rec_value:.2f}/100.00,
which falls mainly under the '{dominant_terms['Recommendation'].capitalize()}' category with a {dominant_values['Recommendation']*100:.1f}% membership.

{len(activated_rules)} rule(s) {activated_rules} were activated to reach this conclusion.
"""

    print(final_output)

    air_quality_desc = {
        'good': "The air quality is good, hence the air is clean and safe.",
        'fair': "The air quality is fair and acceptable, but sensitive groups should still be cautious.",
        'unhealthy': "The air quality is unhealthy and can cause health effects for the general public.",
        'hazardous': "The air quality is hazardous and it is a serious health risk for everyone."
    }

    recommendation_desc = {
        'safe outdoor activities': "Everyone is encouraged to perform outdoor activities as usual.",
        'limit outdoor activities': "Limit prolonged outdoor activities and stay indoors when possible.",
        'stay indoors': "Avoid outdoor activities entirely and remain indoors until conditions improve."
    }

    print("Conclusion:")
    print(air_quality_desc[dominant_terms['Air Quality'].lower()])
    print(recommendation_desc[dominant_terms['Recommendation'].lower()])
    print()

    plt.show()
