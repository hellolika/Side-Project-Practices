CREATE PROCEDURE [dbo].[HRMS_RegisterLeaveAmount.1.0.0]
	@memberId INT
AS
BEGIN 
	SET NOCOUNT ON;

	--CHECKING IF MEMBER IS NOT EXISTS
	IF NOT EXISTS(SELECT TOP 1 1 FROM [dbo].[Member] WHERE [Id] = @memberId AND ([IsDeleted] = 0 OR [IsDeleted] IS NULL))
		BEGIN 
			SELECT 3 AS ErrorCode;
			RETURN;
		END

	--CHECKING FOR EXSITING LEAVE AMOUNT
	IF EXISTS(SELECT * FROM [dbo].[LeaveAmount] WHERE [MemberId] = @memberId AND [LeaveType] != 0)
		BEGIN 
			SELECT 41 AS ErrorCode;
			RETURN;
		END

	DECLARE @currentYear AS INT = YEAR(GETDATE());

	INSERT INTO [dbo].[LeaveAmount](
		[MemberId], [LeaveType], [RemainingLeaves], [LeavesGranted], [Year]
	)
	SELECT @memberId, lt.TypeId,
	CASE WHEN lt.TypeId = 1 THEN 1 ELSE lt.DefaultLeavesGranted END,
	CASE WHEN lt.TypeId = 1 THEN 1 ELSE lt.DefaultLeavesGranted END,
	@currentYear
	FROM [dbo].[LeaveType] lt WITH(NOLOCK)
	WHERE lt.[IsEnable] = 1 AND lt.[TypeId] != 0

	SELECT 0 AS ErrorCode;
END