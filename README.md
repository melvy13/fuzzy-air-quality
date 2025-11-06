# A Fuzzy System for Assessing Urban Air Quality Towards Sustainable Cities

This repository contains the Python implementation code for CSC3034 Computational Intelligence - Assignment 1. It is an implementation of a fuzzy inference system using the `scikit-fuzzy` Python library.

The focus of this project is on SDG 11: Sustainable Cities and Communities. The fuzzy system aims to assess the severity of air quality condition in urban areas caused by pollutant concentration (PM2.5), temperature and humidity. Unlike fixed numerical indicators like the AQI that causes sudden changes in categories (e.g. AQI 50 -> Good, AQI 51 -> Moderate), this system allows a more gradual transition between the categories to better evaluate the air quality.

The system takes in 3 input variables:
- Pollutant Concentration (µg/m^3)
- Temperature (°C)
- Humidity (%)

...and produces two output variables:
- Urban Air Quality Condition
- Recommendations

The image below shows the basic structure of the fuzzy system.

<img width="927" height="292" alt="image" src="https://github.com/user-attachments/assets/664935c6-be55-4f55-9ae1-bf0adc16801f" />

## How to Use The Code

At line 139 onwards in `assignment_1.py` are test cases that can be input with different values to represent different scenarios. Each test case is input as a tuple in the form: `(Description, Pollutant Concentration, Temperature, Humidity)`.

For example: `("A perfect day", 2, 25, 50)` represents a perfect day scenario with a PM2.5 pollutant concentration of 2 µg/m^3, a temperature of 25°C, and a relative humidity of 50%.

Input as many test cases as you want and run the code - It will first produce 2 graphs in a separate window. These graphs visualizes the final output of the fuzzy system, one for air quality and one for recommendations. In the terminal are also the detailed text outputs for the test case. Closing both of the graphs (windows) will continue the execution of the code with the next test case, until all test case are run.

## Output Format

Here is an example of how the output would look for the above example.

```
====================
   INPUT ANALYSIS   
====================
Test situation: A perfect day
Pollutant Concentration: 2 µg/m3; Temperature: 25°C; Humidity: 50%

--- Pollutant Concentration Memberships ---
LOW: 0.933
MODERATE: 0.000
HIGH: 0.000
VERY HIGH: 0.000

--- Temperature Memberships ---
COLD: 0.000
MILD: 1.000
HOT: 0.000

--- Humidity Memberships ---
DRY: 0.000
COMFORTABLE: 0.750
HUMID: 0.000

=====================
   RULE ACTIVATION
=====================
Rule 1 activated [Strength: 0.7500]:
  IF (pollutant[low] AND temperature[mild]) AND humidity[comfortable], THEN [air_quality[good], recommendation[safe outdoor activities]]

... (more rules will be listed if they are activated)

=====================
   OUTPUT ANALYSIS
=====================
Final Air Quality Risk Score: 17.34
Final Recommendation Risk Score: 17.34
--- Air Quality Memberships ---
GOOD: 1.000
FAIR: 0.000
UNHEALTHY: 0.000
HAZARDOUS: 0.000

Dominant Category: GOOD (1.000)

--- Recommendation Memberships ---
SAFE OUTDOOR ACTIVITIES: 1.000
LIMIT OUTDOOR ACTIVITIES: 0.000
STAY INDOORS: 0.000

Dominant Category: SAFE OUTDOOR ACTIVITIES (1.000)

====================
   FINAL RESULTS   
====================

Based on the current conditions:
- Pollutant concentration: 2 µg/m3
- Temperature: 25°C
- Humidity: 50%

The urban air quality risk score was evaluated to be 17.34/100.00,
which falls mainly under the 'Good' category with a 100.0% membership.

Meanwhile, the suggested recommendation risk score was evaluated to be 17.34/100.00,
which falls mainly under the 'Safe outdoor activities' category with a 100.0% membership.

1 rule(s) [1] were activated to reach this conclusion.

Conclusion:
The air quality is good, hence the air is clean and safe.
Everyone is encouraged to perform outdoor activities as usual.

```

Have fun with the code ;)

Remember to install dependencies: `pip install numpy scikit-fuzzy matplotlib`
