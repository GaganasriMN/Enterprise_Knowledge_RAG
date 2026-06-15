CREATE TABLE enterprise_records (
  record_id TEXT,
  title TEXT,
  department TEXT,
  category TEXT,
  sensitivity TEXT,
  allowed_roles TEXT,
  summary TEXT
);

INSERT INTO enterprise_records VALUES
('SQL-501', 'Vendor Risk Review', 'Compliance', 'third_party_risk', 'restricted', 'Auditor|Admin', 'Vendor payment processor passed SOC 2 review but requires quarterly evidence checks for encryption and incident notification controls.');

INSERT INTO enterprise_records VALUES
('SQL-610', 'Cloud Cost Optimization', 'Finance', 'cost_control', 'confidential', 'Finance|Admin', 'Reserved instance coverage improved by 18 percent and projected monthly savings are 42000 USD after workload right-sizing.');

INSERT INTO enterprise_records VALUES
('SQL-720', 'Service Ownership Matrix', 'IT', 'operations', 'internal', 'IT|Auditor|Admin', 'Payment API is owned by Platform Engineering with Finance as business owner and Compliance as audit stakeholder.');

INSERT INTO enterprise_records VALUES
('SQL-830', 'SecureConnect Migration Note', 'IT', 'support_operations', 'internal', 'Employee|IT|Admin', 'Legacy VPN Classic references should be mapped to SecureConnect support procedures for current troubleshooting.');

INSERT INTO enterprise_records VALUES
('SQL-915', 'Quarterly Access Review Control', 'Compliance', 'audit_control', 'restricted', 'Auditor|Admin', 'Quarterly entitlement reviews must include reviewer identity, approval status, completion date, and exception remediation evidence.');
