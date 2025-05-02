
### FlightState
- If `'Sky'` and `'Airplane'` in Labels → FlightState = `In-Flight`
- If `'Flying'` in Labels → FlightState = `In-Flight`
- If `'Runway'` OR `'Taxiway'` OR `'Airport'` and `'Airplane'` in Labels → FlightState = `Taxiing`
- If `'Airplane'` in Labels and `'Sky'` and `'Flying'` not in Labels → FlightState = `Grounded`

### TimeOfDay
- If `'Night'` in Labels OR DetectedText → TimeOfDay = `Night`
- If `'Sunset'` in Labels OR DetectedText → TimeOfDay = `Sunset`
-  If `'Sunrise'` in Labels OR DetectedText → TimeOfDay = `Sunrise`
- Else → TimeOfDay = `Day`

### Weather
- If `'Snow'` or `'Snowstorm'` in Labels → Weather = `Snowy`
- If `'Rain'` or `'Storm'` or `'Rainstorm'` in Labels → Weather = `Rainy`
- If `'Clouds'` or `'Overcast'` in Labels → Weather = `Cloudy`
- Else → Weather = `Clear`