# CARMEN-I

CARMEN-I is a corpus of 2,000 clinical records. It consists of discharge letters, referrals and radiology reports generated at the Hospital Clínic of Barcelona (HCB) from March 2020 to March 2022. The reports are written mainly in Spanish, with some sections in Catalan. The corpus covers patients admitted with COVID-19, and includes a wide variety of comorbidities, such as kidney failure, chronic cardiovascular and respiratory diseases, malignancies and immunosuppression. CARMEN-I has been exhaustively anonymized and validated by hospital physicians, NLP experts and linguists, following detailed annotation guidelines, and replacing original sensitive data elements by synthetic equivalents. A subset of the corpus has been annotated with key medical concepts labeled by experts, namely, symptoms, diseases, procedures, medications, species and humans (incl. family members).

CARMEN-I serves as a resource for training and evaluation of clinical natural language processing (NLP) approaches and language models, including tasks such as text de-identification and anonymization of sensitive data; automatic detection of clinical concepts, etc.

CARMEN-I has been created by the Barcelona Supercomputing Center (BSC)'s NLP for Biomedical Information Analysis (NLP4BIA) team in collaboration with Barcelona's Hospital Clínic and the Universitat de Barcelona's CLiC group.

## Files and Folder Structure

- `txt/`
Contains the CARMEN-I text files in two versions: with masked sensitive data (e.g. '01/01/2020' becomes 'FECHAS'; `masked/` folder) and replaced sensitive data (e.g. '01/01/2020' becomes '03/07/2013'; `replaced/` folder).

- `ann/`
Contains the CARMEN-I entity annotations in brat's standalone `.ann` format. Again, there is a different folder for each anonymized version. On top of that, sensitive items annotations (`anon/` folder) and medical named entity annotations (`ner/` folder) are given separately. All medical entity annotations are given together, if you wish to work with just certain labels (e.g. only diseases) you need to separate them yourself. Additionally, the brat configuration files (`annotation.conf` and `visual.conf`) are also provided.

For more information about `.ann` format please visit [brat's website](https://brat.nlplab.org/standoff.html).

- `tsv/`
Contains the CARMEN-I entity annotations in `.tsv` format. Again, there is a different folder for each anonymized version. On top of that, sensitive items annotations and medical named entity annotations are given separately.

Each `.tsv` file contains the following columns: name (associated filename), tag (annotation label), span (start and end character position in text), text (annotation content).

Additionally, the dataset includes a file called `CARMEN1_mappings.tsv`, in which every file is classified in two aspects: its language (`es` for Spanish, `cat` for Catalan, `bi` for bilingual texts that include a mix of both languages) and whether it has clinical concept recognition annotations (either `True` or `False`).

## License

CARMEN-I is available under Creative Commons Attribution-ShareAlike 4.0 International Public Licenses (CC-BY-SA).

Users of CARMEN–I must register and provide information about their intended use of the resource. The purpose is to better know the use of the resource and to inform the patients about the current use of shared data. Users must acknowledge the access conditions, including the license, permissions, restrictions, obligations, Data Protection Agreement (DPA), and disclaimer. Users must also agree to use the resource only for its intended purpose and to maintain the anonymization of the data. The license allows users to share, adapt, and build upon the resource for any purpose, including commercial uses, as long as appropriate credit is given and modifications are indicated.

If the user detects any expression with suspected possible identification, it is their obligation to immediately notify the CARMEN-I authors at infosic@clinic.cat.

## Contact

For any questions or suggestions, please contact:

- Martin Krallinger (BSC's NLP for Biomedical Information Analysis Head | krallinger.martin@gmail.com)
- Salvador Lima (BSC's NLP for Biomedical Information Analysis | salvador.limalopez@gmail.com)
- Xavier Borrat (Barcelona's Hospital Clínic Health Informatics Dpt. Head | xborrat@mit.edu)
