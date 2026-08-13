CREATE PROCEDURE [dbo].[HRMS_EditMemberLeaves.1.0.0]
	@memberid int,
	@leaveTypeId int, 
	@readjustAmount DECIMAL(16,9)
AS
BEGIN
	--CHECKING IF USER IS EXIST
	IF((SELECT TOP 1 1 FROM [dbo].[Member] WHERE [Id] = @memberId) IS NULL)
		BEGIN
			--USER NOT FOUND ERROR
			SELECT 32 AS ErrorCode
			RETURN;
		END
	--CHECKING IF LEAVE TYPE ID IS EXIST
	IF((SELECT TOP 1 1 FROM [dbo].[LeaveType] WHERE [TypeId] = @leaveTypeId) IS NULL)
		BEGIN
			--INVALID LEVE TYPE ID ERORR
			SELECT 13 AS ErrorCode
			RETURN;
		END

	--CHECKING IF USER IS STILL IN PROBATION
	IF((SELECT [IsInProbation] FROM [dbo].[Member] WHERE [Id] = @memberid) = 1)
		BEGIN
			--MEMBER STILL IN PROBATION ERROR
			SELECT 33 AS ErrorCode
			RETURN;
		END

	--PROCEED TO READJUST THE LEAVE REMAINING OF MEMBER
	UPDATE [dbo].[LeaveAmount]
	SET [RemainingLeaves] = [RemainingLeaves] + @readjustAmount,
		[LeavesGranted] = [LeavesGranted] + @readjustAmount
	WHERE [LeaveType] = @leaveTypeId AND [MemberId] = @memberid;
	SELECT 0 AS	ErrorCode;
END
