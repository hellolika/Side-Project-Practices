CREATE PROCEDURE [dbo].[HRMS_DeleteMember]
  @memberId INT
AS
BEGIN
  UPDATE Member
  SET IsDeleted = 1
  WHERE Id = @memberId;
  

  SELECT 0 AS ErrorCode;
END
