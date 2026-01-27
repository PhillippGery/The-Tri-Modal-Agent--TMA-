# The Tri-Modal Agent (TMA):
# A LLaVA-Based Agent for Chained Perception and Conditional Generation


Developing general-purpose multimodal agents requires
seamlessly integrating large language models (LLMs) with
specialized vision foundation models (FMs). This paper
presents the Tri-Modal Agent (TMA) framework, a mod-
ular architecture that translates complex, multi-step natu-
ral language instructions into sequential, chained image-
to-image operations. Inspired by LLaVA’s agentic reason-
ing, the TMA leverages the capabilities of SAM for prompt-
able perception and ControlNet for conditional genera-
tion. It utilizes an Agent Scheduler to orchestrate data
flow between these disparate models. A substantial evalua-
tion confirms that the architecture is 100% reliable in tool
chaining and memory management. These results validate
the TMA Framework’s effectiveness as a robust, memory-
optimized solution for executing complex multimodal tasks
on resource-constrained platforms. It provides clear proof
of concept for the future of composable foundation models
in a scalable environment.
