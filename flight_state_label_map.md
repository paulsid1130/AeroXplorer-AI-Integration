# Flight State Label Mapping Rules

We classify images into flight states based on the presence of specific labels detected by Amazon Rekognition:

- **Taxiing**:  
  If any of the following labels are present: `Runway`, `Airport`, `Tarmac`.

- **In-Flight**:  
  If any of the following labels are present: `Flying`, `Sky`, `Airplane`.

- **Grounded**:  
  If any of the following labels are present: `Gate`, `Terminal`, `Hangar`, `Parking`, `Jet Bridge`.

If none of the above labels are matched, classify the image as **Unknown**.
