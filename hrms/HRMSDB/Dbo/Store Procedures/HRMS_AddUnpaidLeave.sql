CREATE PROCEDURE [dbo].[HRMS_AddUnpaidLeave.1.0.0]
	@memberId int = 0
AS
BEGIN
	SET NOCOUNT ON;

	DECLARE @currentYear AS INT = YEAR(GETDATE());
	
	IF((SELECT TOP 1 1 FROM [dbo].[Member] WHERE [Id] = @memberId) IS NOT NULL)
		BEGIN
		INSERT INTO [dbo].[LeaveAmount] (
			[MemberId], [LeaveType], [RemainingLeaves], [LeavesGranted], [Year]
		)
		VALUES(@memberId, 0, 0, 0,@currentYear);
		SELECT 0 AS ErrorCode;
		END
	ELSE
	SELECT 32 AS ErrorCode;
END
