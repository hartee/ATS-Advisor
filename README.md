# ATS-Advisor
Service to run resumes through ATS scoring

The service as it currently is built is pretty bare bones and almost useless, really.

It does a cosine-similarity match between a resume and a job description, but that isn't really a good measure.

The suggestions are rubbish too.  The basic problem is that it doesn't know what words actually have meaning in a job description and which ones are just fluff.
