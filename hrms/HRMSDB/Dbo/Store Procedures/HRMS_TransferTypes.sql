CREATE PROCEDURE [dbo].[HRMS_TransferTypes.1.0.0]
AS
BEGIN
  Select Id, TransferName , BeneficiaryTypeId, IsEnable from TransferType
END
