# TechLead-AI-Assignment
Assignments for Tech Lead AI role

Execution Steps (bash):

1) install requirements

2) Clean Docker resources
   docker system prune -a -f

3) Build the lite container
   docker build -t api-lite -f Dockerfile.lite .

4) Run container
   docker run -p 5001:5000 api-lite

5) In another terminal, run the test script (and install matplotlib if not already)
   pip install matplotlib requests pillow
   python test_lite.py
   
