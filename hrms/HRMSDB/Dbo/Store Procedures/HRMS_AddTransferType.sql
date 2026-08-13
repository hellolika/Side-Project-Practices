Create PROCEDURE [dbo].[HRMS_AddTransferType.1.0.0]
  @transferName NVARCHAR(100),
  @isEnable BIT,
  @createdBy INT,
  @modifiedBy INT
AS
BEGIN
  INSERT INTO [dbo].[TransferType] ([TransferName], [IsEnable], [CreatedBy], [ModifiedBy]) 
  VALUES (@transferName, @isEnable, @createdBy, @modifiedBy);

  SELECT 0 AS ErrorCode;
END
