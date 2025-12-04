class SimpleRAG:
    """Very simple RAG: holds a few predefined context chunks and returns them."""

    def __init__(self):
        self.docs = [
            # Context 1 – IoT security + STRIDE + IAM
            """IoT devices face several security challenges: limited computing power, insecure firmware, hard-coded passwords,
weak network isolation, and absence of software updates. Common attacks include device spoofing, tampering, man-in-the-middle,
replay attacks, denial of service, and remote code execution.

Threat modelling in IoT can use the STRIDE framework: Spoofing, Tampering, Repudiation, Information Disclosure,
Denial of Service, and Elevation of Privilege.

Identity and access management for IoT devices uses cryptographic keys, device certificates, and mutual authentication
between device and backend. Common access control models are role-based access control (RBAC), attribute-based access control (ABAC),
and capability-based access control. Policies often need to be fine-grained: time-based access, location-based access,
and revocation of compromised identities.""",

            # Context 2 – Blockchain access control
            """Blockchain-based access control uses smart contracts to define and enforce which users or devices can perform which
operations on IoT resources. The blockchain provides decentralization, tamper-resistant logs, and transparency.

A typical architecture includes a Device Registry Contract that stores device identifiers and owners, an Access Policy Contract
that defines RBAC or ABAC rules, and an Audit Log Contract that records each access event.

RBAC on blockchain maps roles to addresses. ABAC can use attributes such as role, deviceType, location, time, and operation.
Capability-based access control can issue signed capability tokens stored or referenced on-chain.

Example policy: the owner address can unlock a smart lock at any time; a guest address can unlock only between 08:00 and 20:00;
a service provider address can unlock only on specific scheduled days.""",

            # Context 3 – ML anomaly detection for IoT
            """Machine learning based anomaly detection for IoT devices monitors network traffic or device behavior to detect compromised
devices or ongoing attacks. Features can include packet rate, protocol distribution, number of distinct IPs, burstiness,
and payload statistics.

Unsupervised models such as autoencoders, Isolation Forest, and clustering can learn normal behavior and flag deviations.
Supervised models can be trained on labeled datasets such as N-BaIoT containing benign and attack traffic for infected IoT devices.

Common attacks detected are scanning, UDP flooding, TCP SYN flooding, ACK flooding, and command-and-control communication.

When an anomaly is detected with high confidence, recommended actions include: isolating the device from the network,
revoking or suspending its access rights in the blockchain access control contract, rotating its cryptographic keys,
forcing a firmware update, and recording an incident in an immutable audit log."""
        ]

    def get_relevant(self, query: str, k: int = 3):
        """Return up to k context chunks (here we just return first k)."""
        return self.docs[:k]
